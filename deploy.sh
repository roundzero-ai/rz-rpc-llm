#!/usr/bin/env bash
# ==============================================================================
# rz-rpc-llm — LLaMA.cpp Deployment
#
# Two modes:
#   - Distributed:  DGX Spark GB10 (CUDA)  → rpc-server  +  Mac Studio (Metal)
#   - Vision:      Mac Studio alone with --vision flag (Qwen3.5 + mmproj)
#
# Usage: ./deploy.sh <command> [options]
#   clone   [--tag TAG]              Clone llama.cpp (`latest` resolves latest tag)
#   build-dgx                        Sync source to DGX and build
#   build-mac                        Build on local Mac
#   download [--repo R] [--pattern P] [--dir D] [--vision]
#   start-rpc                        Start rpc-server on DGX
#   stop-rpc                         Stop rpc-server on DGX
#   start-llama [--model-file F] [--alias A] [--ctx N] [--parallel N] [--vision]
#   stop-llama                       Stop local llama-server
#   start-monitor-web [--port N]     Start one-port API + monitor gateway
#   stop-monitor-web                 Stop one-port API + monitor gateway
#   deploy  [--tag TAG] [--model-file F] [--alias A]
#   full    [--tag TAG] [--model-file F] [--alias A]
#                                    Full pipeline: clone→build→start
#   status                           Show running processes
#   logs    [rpc|llama]              Tail logs
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULTS_FILE="${SCRIPT_DIR}/defaults.env"
CONFIG_FILE="${SCRIPT_DIR}/config.env"
PID_DIR="${SCRIPT_DIR}/.pids"
LOG_DIR="${SCRIPT_DIR}/logs"

# ------------------------------------------------------------------------------
# Colours & logging
# ------------------------------------------------------------------------------
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; MAGENTA='\033[0;35m'
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
    local env_dgx_host="${DGX_HOST-}"
    local env_dgx_user="${DGX_USER-}"
    local env_dgx_remote_dir="${DGX_REMOTE_DIR-}"
    local env_hf_token="${HF_TOKEN-}"

    if [[ ! -f "${DEFAULTS_FILE}" ]]; then
        die "defaults.env not found. Re-clone the repo or restore the checked-in defaults file."
    fi

    # shellcheck source=/dev/null
    source "${DEFAULTS_FILE}"
    log "Loaded defaults: ${DEFAULTS_FILE}"

    if [[ -f "${CONFIG_FILE}" ]]; then
        # Snapshot default values before applying local overrides.
        # After sourcing config.env we compare and warn about every override
        # so that stale or unintentional changes are visible at startup.
        local _drift_keys=()
        local _drift_snap=()
        while IFS='=' read -r _key _; do
            [[ -z "${_key}" || "${_key}" == \#* ]] && continue
            _key="${_key%%[[:space:]]*}"
            _drift_keys+=("${_key}")
            _drift_snap+=("${!_key:-}")
        done < <(grep -E '^[A-Z_]+=' "${CONFIG_FILE}")

        # shellcheck source=/dev/null
        source "${CONFIG_FILE}"
        log "Loaded local overrides: ${CONFIG_FILE}"

        # Report every value that config.env changed from the checked-in default
        local _n_overrides=0
        for _i in "${!_drift_keys[@]}"; do
            local _k="${_drift_keys[${_i}]}"
            local _default="${_drift_snap[${_i}]}"
            local _current="${!_k:-}"
            if [[ "${_current}" != "${_default}" ]]; then
                log_warn "config.env override: ${_k}=${_current}  (default: ${_default})"
                (( _n_overrides++ ))
            fi
        done
        if (( _n_overrides > 0 )); then
            log_warn "${_n_overrides} override(s) detected — review config.env if unexpected"
        fi
    else
        log "No local override file at ${CONFIG_FILE} (using checked-in defaults only)"
    fi

    [[ -n "${env_dgx_host}" ]] && DGX_HOST="${env_dgx_host}"
    [[ -n "${env_dgx_user}" ]] && DGX_USER="${env_dgx_user}"
    [[ -n "${env_dgx_remote_dir}" ]] && DGX_REMOTE_DIR="${env_dgx_remote_dir}"
    [[ -n "${env_hf_token}" ]] && HF_TOKEN="${env_hf_token}"

    # Expand relative paths relative to SCRIPT_DIR
    if [[ "${LLAMA_CPP_DIR}" != /* ]]; then
        LLAMA_CPP_DIR="${SCRIPT_DIR}/${LLAMA_CPP_DIR#./}"
    fi
    if [[ "${MODELS_DIR}" != /* ]]; then
        MODELS_DIR="${SCRIPT_DIR}/${MODELS_DIR#./}"
    fi

    # Backward-compatible defaults for newer tuning flags
    LLAMA_CACHE_PROMPT="${LLAMA_CACHE_PROMPT:-0}"
    LLAMA_CACHE_RAM="${LLAMA_CACHE_RAM:-0}"
    LLAMA_CONT_BATCHING="${LLAMA_CONT_BATCHING:-1}"
    LLAMA_NO_CONTEXT_SHIFT="${LLAMA_NO_CONTEXT_SHIFT:-0}"

    # Mode-specific model patterns and runtime defaults
    DEFAULT_VISION_MODEL_PATTERN="${DEFAULT_VISION_MODEL_PATTERN:-${DEFAULT_MODEL_PATTERN:-*UD-Q6_K_XL*}}"

    LLAMA_TEXT_CTX_SIZE="${LLAMA_TEXT_CTX_SIZE:-${LLAMA_CTX_SIZE:-131072}}"
    LLAMA_TEXT_PARALLEL="${LLAMA_TEXT_PARALLEL:-${LLAMA_PARALLEL:-1}}"
    LLAMA_TEXT_BATCH_SIZE="${LLAMA_TEXT_BATCH_SIZE:-${LLAMA_BATCH_SIZE:-2048}}"
    LLAMA_TEXT_UBATCH_SIZE="${LLAMA_TEXT_UBATCH_SIZE:-${LLAMA_UBATCH_SIZE:-512}}"
    LLAMA_TEXT_N_GPU_LAYERS="${LLAMA_TEXT_N_GPU_LAYERS:-${LLAMA_N_GPU_LAYERS:-all}}"
    LLAMA_TEXT_CACHE_TYPE_K="${LLAMA_TEXT_CACHE_TYPE_K:-${LLAMA_CACHE_TYPE_K:-q8_0}}"
    LLAMA_TEXT_CACHE_TYPE_V="${LLAMA_TEXT_CACHE_TYPE_V:-${LLAMA_CACHE_TYPE_V:-q8_0}}"
    LLAMA_TEXT_TEMP="${LLAMA_TEXT_TEMP:-1.0}"
    LLAMA_TEXT_TOP_P="${LLAMA_TEXT_TOP_P:-0.95}"
    LLAMA_TEXT_TOP_K="${LLAMA_TEXT_TOP_K:-40}"
    LLAMA_TEXT_MIN_P="${LLAMA_TEXT_MIN_P:-0.01}"
    LLAMA_TEXT_REPEAT_PENALTY="${LLAMA_TEXT_REPEAT_PENALTY:-1.0}"
    LLAMA_TEXT_PRESENCE_PENALTY="${LLAMA_TEXT_PRESENCE_PENALTY:-0.0}"
    LLAMA_TEXT_REASONING_FORMAT="${LLAMA_TEXT_REASONING_FORMAT:-auto}"

    LLAMA_VISION_CTX_SIZE="${LLAMA_VISION_CTX_SIZE:-${LLAMA_CTX_SIZE:-262144}}"
    LLAMA_VISION_PARALLEL="${LLAMA_VISION_PARALLEL:-${LLAMA_PARALLEL:-1}}"
    LLAMA_VISION_BATCH_SIZE="${LLAMA_VISION_BATCH_SIZE:-${LLAMA_BATCH_SIZE:-2048}}"
    LLAMA_VISION_UBATCH_SIZE="${LLAMA_VISION_UBATCH_SIZE:-${LLAMA_UBATCH_SIZE:-512}}"
    LLAMA_VISION_N_GPU_LAYERS="${LLAMA_VISION_N_GPU_LAYERS:-${LLAMA_N_GPU_LAYERS:-all}}"
    LLAMA_VISION_CACHE_TYPE_K="${LLAMA_VISION_CACHE_TYPE_K:-${LLAMA_CACHE_TYPE_K:-q8_0}}"
    LLAMA_VISION_CACHE_TYPE_V="${LLAMA_VISION_CACHE_TYPE_V:-${LLAMA_CACHE_TYPE_V:-q8_0}}"
    LLAMA_VISION_TEMP="${LLAMA_VISION_TEMP:-${LLAMA_TEMP:-1.0}}"
    LLAMA_VISION_TOP_P="${LLAMA_VISION_TOP_P:-${LLAMA_TOP_P:-0.95}}"
    LLAMA_VISION_TOP_K="${LLAMA_VISION_TOP_K:-${LLAMA_TOP_K:-20}}"
    LLAMA_VISION_MIN_P="${LLAMA_VISION_MIN_P:-${LLAMA_MIN_P:-0.0}}"
    LLAMA_VISION_REPEAT_PENALTY="${LLAMA_VISION_REPEAT_PENALTY:-${LLAMA_REPEAT_PENALTY:-1.0}}"
    LLAMA_VISION_PRESENCE_PENALTY="${LLAMA_VISION_PRESENCE_PENALTY:-${LLAMA_PRESENCE_PENALTY:-1.5}}"
    LLAMA_VISION_REASONING_FORMAT="${LLAMA_VISION_REASONING_FORMAT:-auto}"

    # Allow HF_TOKEN override from environment
    HF_TOKEN="${HF_TOKEN:-}"
    if [[ -n "${HF_TOKEN_ENV:-}" ]]; then
        HF_TOKEN="${HF_TOKEN_ENV}"
    fi

    LLAMA_BACKEND_HOST="${LLAMA_BACKEND_HOST:-127.0.0.1}"
    LLAMA_BACKEND_PORT="${LLAMA_BACKEND_PORT:-8682}"
    MONITOR_WEB_HOST="${MONITOR_WEB_HOST:-${LLAMA_HOST}}"
    MONITOR_WEB_PORT="${MONITOR_WEB_PORT:-${LLAMA_PORT}}"
    LLAMA_DEFRAG_THOLD="${LLAMA_DEFRAG_THOLD:-0.05}"
}

require_dgx_config() {
    [[ -n "${DGX_HOST:-}" ]] || die "DGX_HOST is empty. Set it in config.env or your shell for distributed mode."
    [[ -n "${DGX_USER:-}" ]] || die "DGX_USER is empty. Set it in config.env or your shell for distributed mode."
    [[ -n "${DGX_REMOTE_DIR:-}" ]] || die "DGX_REMOTE_DIR is empty. Set it in config.env or your shell for distributed mode."
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

resolve_latest_local_tag() {
    local repo_dir="$1"
    local latest_tag

    latest_tag="$(git -C "${repo_dir}" tag --sort=version:refname | grep '^b[0-9]' | tail -1)"
    if [[ -z "${latest_tag}" ]]; then
        latest_tag="$(git -C "${repo_dir}" tag --sort=version:refname | tail -1)"
    fi
    [[ -n "${latest_tag}" ]] || die "No tags found in ${repo_dir}"

    echo "${latest_tag}"
}

select_runtime_profile() {
    local vision_mode="${1:-}"
    local ctx_override="${2:-}"
    local parallel_override="${3:-}"

    if [[ -n "${vision_mode}" ]]; then
        RUNTIME_PROFILE_NAME="Qwen3.5-122B-A10B"
        RUNTIME_PROFILE_KIND="vision"
        RUNTIME_CTX_SIZE="${ctx_override:-${LLAMA_VISION_CTX_SIZE}}"
        RUNTIME_PARALLEL="${parallel_override:-${LLAMA_VISION_PARALLEL}}"
        RUNTIME_BATCH_SIZE="${LLAMA_VISION_BATCH_SIZE}"
        RUNTIME_UBATCH_SIZE="${LLAMA_VISION_UBATCH_SIZE}"
        RUNTIME_N_GPU_LAYERS="${LLAMA_VISION_N_GPU_LAYERS}"
        RUNTIME_CACHE_TYPE_K="${LLAMA_VISION_CACHE_TYPE_K}"
        RUNTIME_CACHE_TYPE_V="${LLAMA_VISION_CACHE_TYPE_V}"
        RUNTIME_TEMP="${LLAMA_VISION_TEMP}"
        RUNTIME_TOP_P="${LLAMA_VISION_TOP_P}"
        RUNTIME_TOP_K="${LLAMA_VISION_TOP_K}"
        RUNTIME_MIN_P="${LLAMA_VISION_MIN_P}"
        RUNTIME_REPEAT_PENALTY="${LLAMA_VISION_REPEAT_PENALTY}"
        RUNTIME_PRESENCE_PENALTY="${LLAMA_VISION_PRESENCE_PENALTY}"
        RUNTIME_REASONING_FORMAT="${LLAMA_VISION_REASONING_FORMAT}"
    else
        RUNTIME_PROFILE_NAME="MiniMax-M2.5"
        RUNTIME_PROFILE_KIND="distributed"
        RUNTIME_CTX_SIZE="${ctx_override:-${LLAMA_TEXT_CTX_SIZE}}"
        RUNTIME_PARALLEL="${parallel_override:-${LLAMA_TEXT_PARALLEL}}"
        RUNTIME_BATCH_SIZE="${LLAMA_TEXT_BATCH_SIZE}"
        RUNTIME_UBATCH_SIZE="${LLAMA_TEXT_UBATCH_SIZE}"
        RUNTIME_N_GPU_LAYERS="${LLAMA_TEXT_N_GPU_LAYERS}"
        RUNTIME_CACHE_TYPE_K="${LLAMA_TEXT_CACHE_TYPE_K}"
        RUNTIME_CACHE_TYPE_V="${LLAMA_TEXT_CACHE_TYPE_V}"
        RUNTIME_TEMP="${LLAMA_TEXT_TEMP}"
        RUNTIME_TOP_P="${LLAMA_TEXT_TOP_P}"
        RUNTIME_TOP_K="${LLAMA_TEXT_TOP_K}"
        RUNTIME_MIN_P="${LLAMA_TEXT_MIN_P}"
        RUNTIME_REPEAT_PENALTY="${LLAMA_TEXT_REPEAT_PENALTY}"
        RUNTIME_PRESENCE_PENALTY="${LLAMA_TEXT_PRESENCE_PENALTY}"
        RUNTIME_REASONING_FORMAT="${LLAMA_TEXT_REASONING_FORMAT}"
    fi
}


# ------------------------------------------------------------------------------
# SSH helpers — ControlMaster so password is entered only once per session
# ------------------------------------------------------------------------------
DGX_SSH_OPTS=(
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=10
    -o ControlMaster=auto
    -o "ControlPath=${PID_DIR}/ssh-dgx.ctl"
    -o ControlPersist=600
)

ssh_dgx() {
    ssh "${DGX_SSH_OPTS[@]}" "${DGX_USER}@${DGX_HOST}" "$@"
}

# Establish SSH master connection — password is prompted here and only here.
# All subsequent ssh_dgx and rsync calls reuse the socket transparently.
check_dgx_ssh() {
    mkdir -p "${PID_DIR}"
    if ssh -O check -o "ControlPath=${PID_DIR}/ssh-dgx.ctl" \
            "${DGX_USER}@${DGX_HOST}" &>/dev/null; then
        log_ok "DGX SSH master already active (${DGX_USER}@${DGX_HOST})"
        return 0
    fi
    log "Connecting to DGX (${DGX_USER}@${DGX_HOST}) — enter password once if prompted..."
    if ! ssh_dgx "echo OK" ; then
        die "Cannot SSH to DGX at ${DGX_USER}@${DGX_HOST}. Check host, user, and password/keys."
    fi
    log_ok "DGX SSH connection established — all subsequent commands reuse this session"
}

# ------------------------------------------------------------------------------
# Command: clone
# ------------------------------------------------------------------------------
cmd_clone() {
    local tag=""
    local repo="https://github.com/ggml-org/llama.cpp.git"
    local resolved_tag=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --tag|-t) tag="$2"; shift 2 ;;
            --repo)   repo="$2"; shift 2 ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    log_section "Clone llama.cpp"
    log "Repo : ${repo}"
    log "Tag  : ${tag:-HEAD}"
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
        resolved_tag="${tag}"
        if [[ "${tag}" == "latest" ]]; then
            resolved_tag="$(resolve_latest_local_tag "${LLAMA_CPP_DIR}")"
            log "Resolved latest RPC-compatible tag: ${resolved_tag}"
        fi

        log "Checking out tag/commit: ${resolved_tag}"
        git -C "${LLAMA_CPP_DIR}" checkout "${resolved_tag}"
        log_ok "Checked out: $(git -C "${LLAMA_CPP_DIR}" describe --always --tags)"
    else
        log_ok "At HEAD: $(git -C "${LLAMA_CPP_DIR}" log -1 --oneline)"
    fi
}

# ------------------------------------------------------------------------------
# Command: build-dgx  (runs entirely on DGX via SSH — Mac has no CUDA)
# ------------------------------------------------------------------------------
cmd_build_dgx() {
    log_section "Build llama.cpp on DGX Spark GB10"
    load_config
    require_dgx_config
    check_dgx_ssh

    [[ -d "${LLAMA_CPP_DIR}/.git" ]] || die "llama.cpp not cloned. Run: ./deploy.sh clone"

    log "Syncing source to ${DGX_USER}@${DGX_HOST}:${DGX_REMOTE_DIR} ..."
    ssh_dgx "mkdir -p '${DGX_REMOTE_DIR}'"
    # Pass the same ControlPath so rsync reuses the master socket (no extra password prompt)
    rsync -az --progress \
        -e "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
               -o ControlMaster=auto \
               -o ControlPath=${PID_DIR}/ssh-dgx.ctl \
               -o ControlPersist=600" \
        --exclude='.git' \
        --exclude='build/' \
        --exclude='*.o' \
        "${LLAMA_CPP_DIR}/" \
        "${DGX_USER}@${DGX_HOST}:${DGX_REMOTE_DIR}/"
    log_ok "Sync complete"

    # Use bash -l (login shell) so the full PATH from .profile/.bashrc is loaded.
    # Non-interactive SSH has a bare PATH — cmake, nvcc, find, etc. are all missing.
    # Unquoted heredoc: ${DGX_REMOTE_DIR} expands on Mac side; \$(nproc) on DGX side.
    log "Building on DGX (login shell, CUDA SM121)..."
    ssh_dgx bash -l << ENDSSH
set -euo pipefail
cd '${DGX_REMOTE_DIR}'
rm -rf build
echo "[DGX] PATH: \$PATH"
echo "[DGX] cmake: \$(which cmake 2>/dev/null || echo NOT FOUND)"
echo "[DGX] nvcc:  \$(which nvcc  2>/dev/null || echo NOT FOUND)"
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
echo "[DGX] building rpc-server target..."
cmake --build build --config Release -j\$(nproc) --target rpc-server
echo "[DGX] Executables built:"
find build -type f -executable ! -name '*.so' | sort | sed 's/^/  /'
ENDSSH
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
    rm -rf build
    log "cmake configure (Metal, RPC)..."
    cmake -B build \
        -DGGML_METAL=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_RPC=ON \
        -DCMAKE_BUILD_TYPE=Release

    local jobs
    jobs=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)
    log "cmake build (${jobs} jobs)..."
    cmake --build build --config Release -j"${jobs}" --target llama-server

    log_ok "Mac build complete. Binaries: $(ls "${LLAMA_CPP_DIR}/build/bin/")"
}

# ------------------------------------------------------------------------------
# Command: download
# ------------------------------------------------------------------------------
cmd_download() {
    local repo=""
    local pattern=""
    local local_dir=""
    local mmproj_pattern=""
    local vision_mode=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo)        repo="$2";       shift 2 ;;
            --pattern|-p)  pattern="$2";    shift 2 ;;
            --dir)         local_dir="$2";   shift 2 ;;
            --vision|-v)   vision_mode="1";  shift ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    load_config
    local token="${HF_TOKEN:-}"
    [[ -z "${token}" ]] && log_warn "HF_TOKEN not set — download may fail for gated models"

    mkdir -p "${MODELS_DIR}"

    if [[ -n "${vision_mode}" ]]; then
        repo="${repo:-${DEFAULT_VISION_REPO:-unsloth/Qwen3.5-122B-A10B-GGUF}}"
        pattern="${pattern:-${DEFAULT_VISION_MODEL_PATTERN}}"
        mmproj_pattern="${DEFAULT_MM_PROJ_PATTERN:-mmproj*.gguf}"
        local_dir="${local_dir:-${MODELS_DIR}}"

        log_section "Download Vision Model"
        log "HF repo  : ${repo}"
        log "Pattern  : ${pattern} + ${mmproj_pattern}"
        log "Local dir: ${local_dir}"

        if ! command -v huggingface-cli &>/dev/null; then
            die "huggingface-cli not found. Install it with: pip install huggingface_hub"
        fi
        log "Downloading model + mmproj (this may take a while)..."
        if [[ -n "${token}" ]]; then
            HF_TOKEN="${token}" huggingface-cli download "${repo}" \
                --local-dir "${local_dir}" \
                --include "${pattern}" "${mmproj_pattern}"
        else
            huggingface-cli download "${repo}" \
                --local-dir "${local_dir}" \
                --include "${pattern}" "${mmproj_pattern}"
        fi

        log_ok "Download complete → ${local_dir}"
        log "GGUF files:"
        find "${local_dir}" -name "*.gguf" | sort | sed 's/^/  /'
        return 0
    fi

    repo="${repo:-${DEFAULT_MODEL_REPO}}"
    pattern="${pattern:-${DEFAULT_MODEL_PATTERN}}"
    local_dir="${local_dir:-${MODELS_DIR}}"

    log_section "Download Model"
    log "HF repo  : ${repo}"
    log "Pattern  : ${pattern}"
    log "Local dir: ${local_dir}"

    if ! command -v huggingface-cli &>/dev/null; then
        die "huggingface-cli not found. Install it with: pip install huggingface_hub"
    fi
    log "huggingface-cli: $(command -v huggingface-cli)"

    log "Downloading (this may take a while)..."
    if [[ -n "${token}" ]]; then
        HF_TOKEN="${token}" huggingface-cli download "${repo}" \
            --local-dir "${local_dir}" \
            --include "${pattern}"
    else
        huggingface-cli download "${repo}" \
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
    require_dgx_config
    check_dgx_ssh

    local log_file="/tmp/rpc-server.log"

    # Find rpc-server binary dynamically — location varies by llama.cpp version
    local rpc_bin
    rpc_bin="$(ssh_dgx "find '${DGX_REMOTE_DIR}/build' -type f -executable -name 'rpc-server' 2>/dev/null | head -1")"
    if [[ -z "${rpc_bin}" ]]; then
        die "rpc-server binary not found anywhere under ${DGX_REMOTE_DIR}/build. Run: ./deploy.sh build-dgx"
    fi
    log "rpc-server binary: ${rpc_bin}"

    log "Checking for existing rpc-server on DGX..."
    if ssh_dgx "pgrep -x rpc-server" &>/dev/null; then
        log_warn "rpc-server already running on DGX ($(ssh_dgx "pgrep -x rpc-server"))"
        log_warn "Run './deploy.sh stop-rpc' first if you want to restart."
        return 0
    fi

    log "Starting rpc-server on DGX (${DGX_HOST}:${DGX_RPC_PORT})..."
    ssh_dgx "
        set -euo pipefail
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
    require_dgx_config
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
    local vision_mode=""
    local monitor_mode=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model-file|-m) model_file="$2";  shift 2 ;;
            --alias|-a)      model_alias="$2"; shift 2 ;;
            --ctx|-c)        ctx_size="$2";    shift 2 ;;
            --parallel|-p)   parallel="$2";    shift 2 ;;
            --vision|-v)     vision_mode="1"; shift ;;
            --monitor)       monitor_mode="1"; shift ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    # Vision mode auto-enables monitor
    [[ -n "${vision_mode}" ]] && monitor_mode="1"

    load_config

    if [[ -n "${vision_mode}" ]]; then
        model_file="${model_file:-${DEFAULT_VISION_MODEL_FILE}}"
        model_alias="${model_alias:-${DEFAULT_VISION_ALIAS}}"
    else
        model_file="${model_file:-${DEFAULT_MODEL_FILE}}"
        model_alias="${model_alias:-${DEFAULT_MODEL_ALIAS}}"
    fi

    local resolved_model resolved_mmproj
    resolved_model="$(resolve_model_file "${model_file}")"
    resolved_mmproj=""
    if [[ -n "${vision_mode}" ]]; then
        resolved_mmproj="$(resolve_model_file "${DEFAULT_VISION_MM_PROJ}")"
    fi

    select_runtime_profile "${vision_mode}" "${ctx_size}" "${parallel}"

    log_section "Start llama-server on Mac Studio"
    log "Model file : ${resolved_model}"
    log "Model alias: ${model_alias}"
    log "Public API : ${LLAMA_HOST}:${LLAMA_PORT}"
    log "Backend    : ${LLAMA_BACKEND_HOST}:${LLAMA_BACKEND_PORT}"

    if [[ -n "${vision_mode}" ]]; then
        log "Mode       : ${GREEN}VISION (multimodal)${RESET}"
        log "MM proj    : ${resolved_mmproj}"
        log "Slots      : ${RUNTIME_PARALLEL}"
        log "KV pool    : ${RUNTIME_CTX_SIZE} (unified, shared across slots)"
    else
        require_dgx_config
        log "Mode       : text-only (RPC: ${DGX_HOST}:${DGX_RPC_PORT})"
        log "Slots      : ${RUNTIME_PARALLEL}"
        log "KV pool    : ${RUNTIME_CTX_SIZE} (unified, shared across slots)"
    fi

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

    if [[ -n "${vision_mode}" ]] && [[ ! -f "${resolved_mmproj}" ]]; then
        die "MM proj file not found: ${resolved_mmproj}. Download with: ./deploy.sh download --vision"
    fi

    log "Profile    : ${RUNTIME_PROFILE_NAME} (${RUNTIME_PROFILE_KIND})"
    log "Sampling   : temp=${RUNTIME_TEMP} top_p=${RUNTIME_TOP_P} top_k=${RUNTIME_TOP_K} min_p=${RUNTIME_MIN_P}"
    log "Penalties  : repeat=${RUNTIME_REPEAT_PENALTY} presence=${RUNTIME_PRESENCE_PENALTY}"
    log "Defrag     : ${LLAMA_DEFRAG_THOLD}"

    log "Launching llama-server..."
    local -a llama_flags=(
        --model            "${resolved_model}"
        --alias            "${model_alias}"
        --jinja
        --temp             "${RUNTIME_TEMP}"
        --top-p            "${RUNTIME_TOP_P}"
        --top-k            "${RUNTIME_TOP_K}"
        --min-p            "${RUNTIME_MIN_P}"
        --repeat-penalty   "${RUNTIME_REPEAT_PENALTY}"
        --presence-penalty "${RUNTIME_PRESENCE_PENALTY}"
        --ctx-size         "${RUNTIME_CTX_SIZE}"
        --host             "${LLAMA_BACKEND_HOST}"
        --port             "${LLAMA_BACKEND_PORT}"
        --prio             "${LLAMA_PRIO}"
        --parallel         "${RUNTIME_PARALLEL}"
        --threads          "${LLAMA_THREADS}"
        --threads-batch    "${LLAMA_THREADS_BATCH}"
        --batch-size       "${RUNTIME_BATCH_SIZE}"
        --ubatch-size      "${RUNTIME_UBATCH_SIZE}"
        --n-gpu-layers     "${RUNTIME_N_GPU_LAYERS}"
        --mmap
        --mlock
        --kv-offload
        --kv-unified
        --flash-attn       "${LLAMA_FLASH_ATTN}"
        --cache-type-k     "${RUNTIME_CACHE_TYPE_K}"
        --cache-type-v     "${RUNTIME_CACHE_TYPE_V}"
        --defrag-thold     "${LLAMA_DEFRAG_THOLD}"
        --perf
        --metrics
    )

    if [[ -n "${RUNTIME_REASONING_FORMAT}" ]]; then
        llama_flags+=(--reasoning-format "${RUNTIME_REASONING_FORMAT}")
    fi

    if [[ -n "${vision_mode}" ]]; then
        llama_flags+=(
            --mmproj     "${resolved_mmproj}"
            --split-mode none
        )
    else
        llama_flags+=(
            --rpc            "${DGX_HOST}:${DGX_RPC_PORT}"
            --split-mode     "${LLAMA_SPLIT_MODE}"
            --tensor-split   "${LLAMA_TENSOR_SPLIT}"
        )
    fi

    if [[ "${LLAMA_CONT_BATCHING}" == "1" ]]; then
        llama_flags+=(--cont-batching)
    fi

    if [[ "${LLAMA_NO_CONTEXT_SHIFT}" == "1" ]]; then
        llama_flags+=(--no-context-shift)
    fi

    if [[ "${LLAMA_CACHE_PROMPT}" == "1" ]]; then
        llama_flags+=(--cache-prompt --cache-reuse "${LLAMA_CACHE_REUSE}")
        if [[ "${LLAMA_CACHE_RAM}" != "0" ]]; then
            llama_flags+=(--cache-ram "${LLAMA_CACHE_RAM}")
        fi
    fi

    nohup "${llama_bin}" \
        "${llama_flags[@]}" \
        > "${log_file}" 2>&1 &
    local llama_pid=$!
    echo "${llama_pid}" > "${pid_file}"

    # Poll health — 2s intervals, 10 min ceiling (large model loads take time)
    log "PID ${llama_pid} — polling health every 2s (up to 10 min)..."
    local elapsed=0
    local limit=600
    while (( elapsed < limit )); do
        if curl -sf "http://127.0.0.1:${LLAMA_BACKEND_PORT}/health" &>/dev/null; then
            echo
            cmd_stop_monitor_web >/dev/null 2>&1 || true
            cmd_start_monitor_web >/dev/null
            log_ok "llama-server ready in ${elapsed}s"
            log_ok "OpenAI-compatible endpoint: http://127.0.0.1:${LLAMA_PORT}/v1"
            log_ok "Monitor UI: http://127.0.0.1:${LLAMA_PORT}/monitor"
            log_ok "Metrics: http://127.0.0.1:${LLAMA_PORT}/metrics"
            log "Log: ${log_file}"
            if [[ -n "${monitor_mode}" ]]; then
                echo
                cmd_monitor
            fi
            return 0
        fi
        if ! kill -0 "${llama_pid}" 2>/dev/null; then
            echo
            log_error "llama-server exited unexpectedly. Last log lines:"
            tail -30 "${log_file}" >&2
            die "llama-server failed to start"
        fi
        printf '\r[%s] Loading model... %ds elapsed (Ctrl+C to abort wait)' \
            "$(date '+%H:%M:%S')" "${elapsed}"
        sleep 2
        (( elapsed += 2 ))
    done
    echo
    log_warn "Server did not respond within ${limit}s — check logs: ${log_file}"
}

# ------------------------------------------------------------------------------
# Command: stop-llama
# ------------------------------------------------------------------------------
cmd_stop_llama() {
    log_section "Stop llama-server"
    load_config
    local pid_file="${PID_DIR}/llama-server.pid"
    local stopped=0

    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}"
            rm -f "${pid_file}"
            log_ok "llama-server (PID ${pid}) stopped"
            stopped=1
        else
            log_warn "PID ${pid} not running, cleaning up stale PID file"
            rm -f "${pid_file}"
        fi
    else
        # Fallback: kill by name
        if pgrep -x llama-server &>/dev/null; then
            pkill -x llama-server && log_ok "llama-server stopped (by name)"
            stopped=1
        else
            log_warn "No llama-server process found"
        fi
    fi

    if [[ "${stopped}" == "1" ]]; then
        cmd_stop_monitor_web >/dev/null 2>&1 || true
    fi
}

# ------------------------------------------------------------------------------
# Command: start-monitor-web
# ------------------------------------------------------------------------------
cmd_start_monitor_web() {
    load_config

    local port_override=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port|-p) port_override="$2"; shift 2 ;;
            *) die "Unknown option: $1" ;;
        esac
    done

    mkdir -p "${PID_DIR}" "${LOG_DIR}"

    local monitor_host="${MONITOR_WEB_HOST}"
    local monitor_port="${port_override:-${MONITOR_WEB_PORT}}"
    local monitor_display_host="${monitor_host}"
    local backend_display_host="${LLAMA_BACKEND_HOST}"
    local pid_file="${PID_DIR}/monitor-web.pid"
    local log_file="${LOG_DIR}/monitor-web.log"
    local app_file="${SCRIPT_DIR}/monitor_web.py"

    [[ -f "${app_file}" ]] || die "monitor_web.py not found at ${app_file}"
    [[ "${monitor_port}" != "${LLAMA_BACKEND_PORT}" ]] || die "MONITOR_WEB_PORT and LLAMA_BACKEND_PORT must differ"

    if [[ -f "${pid_file}" ]]; then
        local old_pid
        old_pid="$(cat "${pid_file}")"
        if kill -0 "${old_pid}" 2>/dev/null; then
            log_warn "monitor-web already running (PID ${old_pid})"
            log_warn "Open: http://${monitor_display_host}:${monitor_port}/monitor"
            return 0
        fi
        rm -f "${pid_file}"
    fi

    if [[ "${monitor_display_host}" == "0.0.0.0" ]]; then
        monitor_display_host="127.0.0.1"
    fi
    if [[ "${backend_display_host}" == "0.0.0.0" ]]; then
        backend_display_host="127.0.0.1"
    fi

    log_section "Start Monitor Web"
    log "Binding    : ${monitor_host}:${monitor_port}"
    log "Endpoint   : http://${monitor_display_host}:${monitor_port}/monitor"

    MONITOR_WEB_HOST="${monitor_host}" \
    MONITOR_WEB_PORT="${monitor_port}" \
    PID_DIR="${PID_DIR}" \
    LOG_DIR="${LOG_DIR}" \
    LLAMA_PORT="${LLAMA_PORT}" \
    LLAMA_BACKEND_HOST="${LLAMA_BACKEND_HOST}" \
    LLAMA_BACKEND_PORT="${LLAMA_BACKEND_PORT}" \
    DGX_HOST="${DGX_HOST:-}" \
    DGX_USER="${DGX_USER:-}" \
    DGX_RPC_PORT="${DGX_RPC_PORT:-50052}" \
    PUBLIC_BASE_URL="http://${monitor_display_host}:${monitor_port}" \
    BACKEND_BASE_URL="http://${backend_display_host}:${LLAMA_BACKEND_PORT}" \
    nohup python3 "${app_file}" --host "${monitor_host}" --port "${monitor_port}" \
        > "${log_file}" 2>&1 &

    local monitor_pid=$!
    echo "${monitor_pid}" > "${pid_file}"

    local elapsed=0
    while (( elapsed < 20 )); do
        if curl -sf "http://${monitor_display_host}:${monitor_port}/api/monitor" &>/dev/null; then
            log_ok "monitor-web ready (PID ${monitor_pid})"
            log_ok "Dashboard: http://${monitor_display_host}:${monitor_port}/monitor"
            log "Log: ${log_file}"
            return 0
        fi
        if ! kill -0 "${monitor_pid}" 2>/dev/null; then
            log_error "monitor-web exited unexpectedly. Last log lines:"
            tail -20 "${log_file}" >&2
            die "monitor-web failed to start"
        fi
        sleep 1
        (( elapsed += 1 ))
    done

    log_warn "monitor-web did not respond yet — check ${log_file}"
}

# ------------------------------------------------------------------------------
# Command: stop-monitor-web
# ------------------------------------------------------------------------------
cmd_stop_monitor_web() {
    load_config
    local pid_file="${PID_DIR}/monitor-web.pid"

    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}"
            rm -f "${pid_file}"
            log_ok "monitor-web (PID ${pid}) stopped"
        else
            log_warn "monitor-web PID ${pid} not running, removing stale PID file"
            rm -f "${pid_file}"
        fi
    else
        log_warn "No monitor-web process found"
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
    log "Tag          : ${tag:-HEAD}"
    log "Model file   : ${model_file:-${DEFAULT_MODEL_FILE}}"
    log "Skip clone   : ${skip_clone}"
    log "Skip build   : ${skip_build}"
    log "Skip download: ${skip_download}"

    if (( ! skip_clone )); then
        local clone_args=()
        [[ -n "${tag}" ]] && clone_args+=(--tag "${tag}")
        cmd_clone "${clone_args[@]+"${clone_args[@]}"}"
    fi

    if (( ! skip_build )); then
        cmd_build_dgx
        cmd_build_mac
    fi

    if (( ! skip_download )); then
        cmd_download
    fi

    log "Stopping existing servers so new build and config take effect..."
    cmd_stop_llama || true
    cmd_stop_rpc || true

    # Start RPC first, then llama-server
    cmd_start_rpc

    local llama_args=()
    [[ -n "${model_file}" ]]  && llama_args+=(--model-file "${model_file}")
    [[ -n "${model_alias}" ]] && llama_args+=(--alias "${model_alias}")
    cmd_start_llama "${llama_args[@]+"${llama_args[@]}"}"

    # Block here until health is confirmed (model may still be loading after start-llama returns)
    log "Waiting for confirmed health check before declaring complete..."
    until curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" &>/dev/null; do
        if ! pgrep -x llama-server &>/dev/null; then
            die "llama-server is no longer running"
        fi
        printf '\r[%s] Still loading...' "$(date '+%H:%M:%S')"
        sleep 5
    done
    echo

    log_section "Deploy Complete"
    log_ok "RPC server : ${DGX_HOST}:${DGX_RPC_PORT}"
    log_ok "LLM server : http://${LLAMA_HOST}:${LLAMA_PORT}"
    log_ok "OpenAI API : http://127.0.0.1:${LLAMA_PORT}/v1"
    log_ok "Metrics   : http://127.0.0.1:${LLAMA_PORT}/metrics"

    cmd_monitor
}

# ------------------------------------------------------------------------------
# Command: monitor  — heartbeat loop, Ctrl+C to stop llama-server
# ------------------------------------------------------------------------------
cmd_monitor() {
    load_config 2>/dev/null || true
    local interval="${1:-30}"
    local table_w="${2:-180}"
    local col_w=12
    local lbl_w=20

    local monitor_mode=""
    if ! ssh -O check -o "ControlPath=${PID_DIR}/ssh-dgx.ctl" \
            "${DGX_USER}@${DGX_HOST}" &>/dev/null 2>&1; then
        monitor_mode="VISION"
    else
        monitor_mode="DISTRIBUTED"
    fi

    log_section "Heartbeat Monitor [${monitor_mode}]"
    log "Rolling table every ${interval}s — press Ctrl+C to stop llama-server"
    echo

    trap '
        echo
        tput cnorm 2>/dev/null || true
        log_warn "Ctrl+C received — stopping llama-server..."
        cmd_stop_llama 2>/dev/null || true
        log_ok "llama-server stopped. Goodbye."
        exit 0
    ' INT TERM

    local -a labels_vision=(
        "Mac RAM used" "Mac GPU util"
        "llama-server" "pp (t/s)" "tg (t/s)"
        "reqs" "prompt tokens" "gen tokens"
    )
    local -a labels_distributed=(
        "Mac RAM used" "Mac GPU util"
        "DGX RAM used" "DGX GPU util"
        "rpc-server" "llama-server"
        "pp (t/s)" "tg (t/s)"
        "reqs" "prompt tokens" "gen tokens"
    )

    local -a labels
    local n_rows
    if [[ "${monitor_mode}" == "VISION" ]]; then
        labels=("${labels_vision[@]}")
        n_rows=8
    else
        labels=("${labels_distributed[@]}")
        n_rows=11
    fi
    local table_h=$(( n_rows + 3 ))  # +3 for header/sep/bottom

    local -a vals_idx
    if [[ "${monitor_mode}" == "VISION" ]]; then
        vals_idx=(0 1 5 6 7 8 9 10)
    else
        vals_idx=(0 1 2 3 4 5 6 7 8 9 10)
    fi

    local -a h_ts=()
    local -a snap=()

    # Reserve vertical space for the table
    local i
    for (( i = 0; i < table_h; i++ )); do echo; done

    tput civis 2>/dev/null || true  # hide cursor during redraw

    while true; do
        local ts; ts="$(date '+%H:%M:%S')"

        # -- Mac memory (all modes) --
        local v0="--"
        local vm_out page_sz total_bytes
        vm_out="$(vm_stat 2>/dev/null || true)"
        page_sz="$(echo "${vm_out}" | awk -F'[()]' '/page size of/{gsub(/[^0-9]/,"",$2); print $2}')"
        total_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
        if [[ -n "${page_sz}" ]] && (( total_bytes > 0 )); then
            local fp sp ip mt mf mu mp
            fp="$(echo "${vm_out}" | awk '/Pages free/{gsub(/\./,""); print $3}')"
            sp="$(echo "${vm_out}" | awk '/Pages speculative/{gsub(/\./,""); print $3}')"
            ip="$(echo "${vm_out}" | awk '/Pages inactive/{gsub(/\./,""); print $3}')"
            mt="$(echo "scale=1; ${total_bytes} / 1073741824" | bc)"
            mf="$(echo "scale=1; (${fp:-0} + ${sp:-0} + ${ip:-0}) * ${page_sz} / 1073741824" | bc)"
            mu="$(echo "scale=1; ${mt} - ${mf}" | bc)"
            mp="$(echo "scale=0; ${mu} * 100 / ${mt}" | bc)"
            v0="$(printf "%.0fG/%d%%" "${mu}" "${mp}")"
        fi

        # -- Mac GPU (all modes) --
        local v1="--"
        local mgu
        mgu="$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null \
            | grep -o '"Device Utilization %"=[0-9]*' \
            | awk -F'=' '{print $NF}' | head -1 || true)"
        [[ -n "${mgu}" ]] && v1="${mgu}%"

        local v2="--" v3="--" v4="--"
        if [[ "${monitor_mode}" == "DISTRIBUTED" ]]; then
            if ssh -O check -o "ControlPath=${PID_DIR}/ssh-dgx.ctl" \
                    "${DGX_USER}@${DGX_HOST}" &>/dev/null 2>&1; then
                local ds=""
                ds="$(ssh_dgx bash -c "'
                    echo \"MEM:\$(free -m 2>/dev/null | awk \"/^Mem:/{printf \\\"%d %d\\\", \\\$3, \\\$2}\")\"
                    echo \"GPU_UTIL:\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d \" \")\"
                    echo \"GPU_MEM:\$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)\"
                    echo \"RPC:\$(pgrep -x rpc-server >/dev/null 2>&1 && echo UP || echo DOWN)\"
                '" 2>/dev/null || true)"
                if [[ -n "${ds}" ]]; then
                    local ml; ml="$(echo "${ds}" | grep '^MEM:' | sed 's/^MEM://')"
                    if [[ -n "${ml}" ]]; then
                        local dm dt
                        read -r dm dt <<< "${ml}"
                        if [[ -n "${dt}" ]] && (( dt > 0 )); then
                            v2="$(printf "%dG/%d%%" "$(( dm / 1024 ))" "$(( dm * 100 / dt ))")"
                        fi
                    fi
                    local gu gm
                    gu="$(echo "${ds}" | grep '^GPU_UTIL:' | sed 's/^GPU_UTIL://')"
                    gm="$(echo "${ds}" | grep '^GPU_MEM:' | sed 's/^GPU_MEM://')"
                    if [[ "${gm}" =~ ^[0-9] ]]; then
                        local mu2 mt2; IFS=', ' read -r mu2 mt2 <<< "${gm}"
                        local gm_gb; gm_gb="$(echo "scale=0; ${mu2} / 1024" | bc)"
                        v3="$(printf "%s%%/%dG" "${gu:-?}" "${gm_gb}")"
                    elif [[ -n "${gu}" ]]; then
                        v3="${gu}%/UMA"
                    fi
                    v4="$(echo "${ds}" | grep '^RPC:' | sed 's/^RPC://')"
                fi
            fi
        fi

        # -- llama-server metrics (all modes) --
        local met pp_val tg_val reqs_val prompt_total pred_total
        v5="DOWN"; v6="--"; v7="--"; v8="--"; v9="--"; v10="--"
        if curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" &>/dev/null; then
            v5="UP"
            met="$(curl -sf "http://127.0.0.1:${LLAMA_PORT}/metrics" 2>/dev/null || true)"
            if [[ -n "${met}" ]]; then
                pp_val="$(echo "${met}" | awk '/^llamacpp:prompt_tokens_seconds /{printf "%.1f", $NF; exit}')"
                [[ -n "${pp_val}" ]] && v6="${pp_val}"
                tg_val="$(echo "${met}" | awk '/^llamacpp:predicted_tokens_seconds /{printf "%.1f", $NF; exit}')"
                [[ -n "${tg_val}" ]] && v7="${tg_val}"
                reqs_val="$(echo "${met}" | awk '/^llamacpp:requests_processing /{printf "%d", $NF; exit}')"
                [[ -n "${reqs_val}" ]] && v8="${reqs_val}"
                prompt_total="$(echo "${met}" | awk '/^llamacpp:prompt_tokens_total /{v=$NF; if(v>=1e6) printf "%.1fM",v/1e6; else if(v>=1e3) printf "%.1fK",v/1e3; else printf "%d",v; exit}')"
                [[ -n "${prompt_total}" ]] && v9="${prompt_total}"
                pred_total="$(echo "${met}" | awk '/^llamacpp:tokens_predicted_total /{v=$NF; if(v>=1e6) printf "%.1fM",v/1e6; else if(v>=1e3) printf "%.1fK",v/1e3; else printf "%d",v; exit}')"
                [[ -n "${pred_total}" ]] && v10="${pred_total}"
            fi
        fi

        # -- Append snapshot --
        h_ts+=("${ts}")
        snap+=("${v0}|${v1}|${v2}|${v3}|${v4}|${v5}|${v6}|${v7}|${v8}|${v9}|${v10}")

        # -- Calculate rolling window --
        local mc total start
        total=${#h_ts[@]}
        mc=$(( (table_w - lbl_w) / (col_w + 1) ))
        (( mc < 1 )) && mc=1
        start=0
        (( total > mc )) && start=$(( total - mc ))

        # -- Draw table (cursor up to overwrite previous) --
        printf "\033[${table_h}A"

        # Header row (timestamps)
        local col_sep; col_sep="$(printf '%*s' "${col_w}" '' | tr ' ' '─')"
        printf "\033[2K${BOLD}%-${lbl_w}s${RESET}" ""
        for (( i = start; i < total; i++ )); do
            printf "│${CYAN}%${col_w}s${RESET}" "${h_ts[$i]}"
        done
        printf "│\n"

        # Separator
        printf "\033[2K%s" "$(printf '%*s' "${lbl_w}" '' | tr ' ' '─')"
        for (( i = start; i < total; i++ )); do
            printf "┼%s" "${col_sep}"
        done
        printf "┤\n"

        # Data rows
        local r val color f
        for (( r = 0; r < n_rows; r++ )); do
            printf "\033[2K${BOLD}%-${lbl_w}s${RESET}" "${labels[$r]}"
            for (( i = start; i < total; i++ )); do
                f="${vals_idx[$r]}"
                val="$(echo "${snap[$i]}" | cut -d'|' -f$((f+1)))"
                color="${RESET}"
                [[ "${val}" == "UP" ]]   && color="${GREEN}"
                [[ "${val}" == "DOWN" ]] && color="${RED}"
                if [[ "${val}" =~ ^[0-9]+\.?[0-9]*G/[0-9]+%$ ]]; then
                    local ram_pct; ram_pct="${val##*/}"; ram_pct="${ram_pct%%%}"
                    if (( ram_pct < 50 )); then
                        color="${GREEN}"
                    elif (( ram_pct < 80 )); then
                        color="${YELLOW}"
                    else
                        color="${RED}"
                    fi
                elif [[ "${labels[$r]}" == "tg (t/s)" && "${val}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                    local tg_val; tg_val="$(echo "${val}" | cut -d'.' -f1)"
                    if (( tg_val > 20 )); then
                        color="${GREEN}"
                    elif (( tg_val < 10 )); then
                        color="${RED}"
                    fi
                elif [[ "${labels[$r]}" == "pp (t/s)" && "${val}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                    local pp_val; pp_val="$(echo "${val}" | cut -d'.' -f1)"
                    if (( pp_val > 400 )); then
                        color="${GREEN}"
                    elif (( pp_val < 250 )); then
                        color="${RED}"
                    fi
                elif [[ "${val}" =~ ^[0-9]+%$ ]]; then
                    local gpu_pct; gpu_pct="${val%\%}"
                    if (( gpu_pct < 30 )); then
                        color="${GREEN}"
                    elif (( gpu_pct < 70 )); then
                        color="${YELLOW}"
                    else
                        color="${RED}"
                    fi
                fi
                printf "│${color}%${col_w}s${RESET}" "${val}"
            done
            printf "│\n"
        done

        # Bottom border
        printf "\033[2K%s" "$(printf '%*s' "${lbl_w}" '' | tr ' ' '─')"
        for (( i = start; i < total; i++ )); do
            printf "┴%s" "${col_sep}"
        done
        printf "┘\n"

        # Reserve space for live log section
        local live_rows=7
        local i
        for (( i = 0; i < live_rows; i++  )); do
            echo
        done
        printf "\033[${live_rows}A"  # Move cursor back to start of live section

        # -- Main loop: rolling table every 30s, live log every 3s --
        local fast_interval=3
        local fast_elapsed=0
        local log_file="${LOG_DIR}/llama-server.log"

        while true; do
            # Every 30s: accumulate a new snapshot for rolling table
            if (( fast_elapsed > 0 )) && (( fast_elapsed % interval == 0 )) || (( fast_elapsed == 0 )); then
                # Check if we need to reinitialize (fast_elapsed == 0 means first time)
                if (( fast_elapsed == 0 )); then
                    # First time: accumulate snapshot, then draw everything
                    :
                else
                    # Get new snapshot and draw rolling table
                    local ts; ts="$(date '+%H:%M:%S')"

                    # Mac memory
                    local v0="--"
                    local vm_out page_sz total_bytes
                    vm_out="$(vm_stat 2>/dev/null || true)"
                    page_sz="$(echo "${vm_out}" | awk -F'[()]' '/page size of/{gsub(/[^0-9]/,"",$2); print $2}')"
                    total_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
                    if [[ -n "${page_sz}" ]] && (( total_bytes > 0 )); then
                        local fp sp ip mt mf mu mp
                        fp="$(echo "${vm_out}" | awk '/Pages free/{gsub(/\./,""); print $3}')"
                        sp="$(echo "${vm_out}" | awk '/Pages speculative/{gsub(/\./,""); print $3}')"
                        ip="$(echo "${vm_out}" | awk '/Pages inactive/{gsub(/\./,""); print $3}')"
                        mt="$(echo "scale=1; ${total_bytes} / 1073741824" | bc)"
                        mf="$(echo "scale=1; (${fp:-0} + ${sp:-0} + ${ip:-0}) * ${page_sz} / 1073741824" | bc)"
                        mu="$(echo "scale=1; ${mt} - ${mf}" | bc)"
                        mp="$(echo "scale=0; ${mu} * 100 / ${mt}" | bc)"
                        v0="$(printf "%.0fG/%d%%" "${mu}" "${mp}")"
                    fi

                    # Mac GPU
                    local v1="--"
                    local mgu
                    mgu="$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null \
                        | grep -o '"Device Utilization %"=[0-9]*' \
                        | awk -F'=' '{print $NF}' | head -1 || true)"
                    [[ -n "${mgu}" ]] && v1="${mgu}%"

                    # llama-server metrics
                    local met pp_val tg_val reqs_val prompt_total pred_total
                    v5="DOWN"; v6="--"; v7="--"; v8="--"; v9="--"; v10="--"
                    if curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" &>/dev/null; then
                        v5="UP"
                        met="$(curl -sf "http://127.0.0.1:${LLAMA_PORT}/metrics" 2>/dev/null || true)"
                        if [[ -n "${met}" ]]; then
                            pp_val="$(echo "${met}" | awk '/^llamacpp:prompt_tokens_seconds /{printf "%.1f", $NF; exit}')"
                            [[ -n "${pp_val}" ]] && v6="${pp_val}"
                            tg_val="$(echo "${met}" | awk '/^llamacpp:predicted_tokens_seconds /{printf "%.1f", $NF; exit}')"
                            [[ -n "${tg_val}" ]] && v7="${tg_val}"
                            reqs_val="$(echo "${met}" | awk '/^llamacpp:requests_processing /{printf "%d", $NF; exit}')"
                            [[ -n "${reqs_val}" ]] && v8="${reqs_val}"
                            prompt_total="$(echo "${met}" | awk '/^llamacpp:prompt_tokens_total /{v=$NF; if(v>=1e6) printf "%.1fM",v/1e6; else if(v>=1e3) printf "%.1fK",v/1e3; else printf "%d",v; exit}')"
                            [[ -n "${prompt_total}" ]] && v9="${prompt_total}"
                            pred_total="$(echo "${met}" | awk '/^llamacpp:tokens_predicted_total /{v=$NF; if(v>=1e6) printf "%.1fM",v/1e6; else if(v>=1e3) printf "%.1fK",v/1e3; else printf "%d",v; exit}')"
                            [[ -n "${pred_total}" ]] && v10="${pred_total}"
                        fi
                    fi

                    h_ts+=("${ts}")
                    snap+=("${v0}|${v1}|${v2}|${v3}|${v4}|${v5}|${v6}|${v7}|${v8}|${v9}|${v10}")
                fi

                # Draw full screen from top (cursor to home)
                printf "\033[H"
                tput civis 2>/dev/null || true  # hide cursor

                # Rolling table (draw in top portion)
                local mc total start
                total=${#h_ts[@]}
                mc=$(( (table_w - lbl_w) / (col_w + 1) ))
                (( mc < 1 )) && mc=1
                start=0
                (( total > mc )) && start=$(( total - mc ))

                # Header row
                local col_sep; col_sep="$(printf '%*s' "${col_w}" '' | tr ' ' '─')"
                printf "${BOLD}%-${lbl_w}s${RESET}" ""
                for (( i = start; i < total; i++ )); do
                    printf "│${CYAN}%${col_w}s${RESET}" "${h_ts[$i]}"
                done
                printf "│\n"

                # Separator
                printf "%s" "$(printf '%*s' "${lbl_w}" '' | tr ' ' '─')"
                for (( i = start; i < total; i++ )); do
                    printf "┼%s" "${col_sep}"
                done
                printf "┤\n"

                # Data rows
                local r val color f
                for (( r = 0; r < n_rows; r++ )); do
                    printf "${BOLD}%-${lbl_w}s${RESET}" "${labels[$r]}"
                    for (( i = start; i < total; i++ )); do
                        f="${vals_idx[$r]}"
                        val="$(echo "${snap[$i]}" | cut -d'|' -f$((f+1)))"
                        color="${RESET}"
                        [[ "${val}" == "UP" ]]   && color="${GREEN}"
                        [[ "${val}" == "DOWN" ]] && color="${RED}"
                        if [[ "${val}" =~ ^[0-9]+\.?[0-9]*G/[0-9]+%$ ]]; then
                            local ram_pct; ram_pct="${val##*/}"; ram_pct="${ram_pct%%%}"
                            if (( ram_pct < 50 )); then
                                color="${GREEN}"
                            elif (( ram_pct < 80 )); then
                                color="${YELLOW}"
                            else
                                color="${RED}"
                            fi
                        elif [[ "${labels[$r]}" == "tg (t/s)" && "${val}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                            local tg_val; tg_val="$(echo "${val}" | cut -d'.' -f1)"
                            if (( tg_val > 20 )); then
                                color="${GREEN}"
                            elif (( tg_val < 10 )); then
                                color="${RED}"
                            fi
                        elif [[ "${labels[$r]}" == "pp (t/s)" && "${val}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                            local pp_val; pp_val="$(echo "${val}" | cut -d'.' -f1)"
                            if (( pp_val > 400 )); then
                                color="${GREEN}"
                            elif (( pp_val < 250 )); then
                                color="${RED}"
                            fi
                        elif [[ "${val}" =~ ^[0-9]+%$ ]]; then
                            local gpu_pct; gpu_pct="${val%\%}"
                            if (( gpu_pct < 30 )); then
                                color="${GREEN}"
                            elif (( gpu_pct < 70 )); then
                                color="${YELLOW}"
                            else
                                color="${RED}"
                            fi
                        fi
                        printf "│${color}%${col_w}s${RESET}" "${val}"
                    done
                    printf "│\n"
                done

                # Bottom border
                printf "%s" "$(printf '%*s' "${lbl_w}" '' | tr ' ' '─')"
                for (( i = start; i < total; i++ )); do
                    printf "┴%s" "${col_sep}"
                done
                printf "┘\n"

                # Reserve live section space
                for (( i = 0; i < live_rows; i++  )); do
                    echo
                done
                fast_elapsed=0
            fi

            # Every 3s: update live log section only (in place)
            sleep "${fast_interval}"
            fast_elapsed=$(( fast_elapsed + fast_interval ))

            if [[ -f "${log_file}" ]]; then
                local last_lines; last_lines="$(tail -5 "${log_file}" 2>/dev/null || true)"
                if [[ -n "${last_lines}" ]]; then
                    local ts_fast; ts_fast="$(date '+%H:%M:%S')"

                    # Calculate position: cursor is at line after rolling table + live_rows
                    # Move cursor to start of live section (table_h lines from top, but rolling table redraws itself)
                    # Since we redrew everything from \033[H, cursor is now at the reserved space
                    # So just clear and overwrite
                    for (( i = 0; i < live_rows; i++  )); do
                        printf "\033[2K\r"  # Clear line and return to start
                        [[ $i -lt $(( live_rows - 1 )) ]] && printf "\033[1A"  # Move up one line
                    done
                    # Now at start of live section

                    printf "${BOLD}%-20s${RESET}" "⚡ live @ ${ts_fast}"
                    printf " %s\r\n" "--- llama-server log (last 5 lines) ---"

                    echo "${last_lines}" | tail -5 | while IFS= read -r line; do
                        printf "${RESET}%-80s${RESET}\r\n" "${line:0:80}"
                    done
                fi
            fi
        done
    done
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
            log_ok "llama-server  RUNNING  (PID ${pid}, backend ${LLAMA_BACKEND_HOST}:${LLAMA_BACKEND_PORT})"
            if curl -sf "http://127.0.0.1:${LLAMA_BACKEND_PORT}/health" &>/dev/null; then
                log_ok "              health check: OK"
            else
                log_warn "              health check: not responding yet"
            fi
        else
            log_warn "llama-server  STOPPED  (stale PID ${pid})"
        fi
    else
        if pgrep -x llama-server &>/dev/null; then
            log_ok "llama-server  RUNNING  (PID $(pgrep -x llama-server | head -1), unmanaged backend)"
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

    local monitor_pid_file="${PID_DIR}/monitor-web.pid"
    local monitor_check_host="${MONITOR_WEB_HOST}"
    [[ "${monitor_check_host}" == "0.0.0.0" ]] && monitor_check_host="127.0.0.1"
    if [[ -f "${monitor_pid_file}" ]]; then
        local monitor_pid
        monitor_pid="$(cat "${monitor_pid_file}")"
        if kill -0 "${monitor_pid}" 2>/dev/null; then
            if curl -sf "http://${monitor_check_host}:${MONITOR_WEB_PORT}/api/monitor" &>/dev/null; then
                log_ok "monitor-web   RUNNING  (PID ${monitor_pid}, http://${monitor_check_host}:${MONITOR_WEB_PORT}/monitor)"
            else
                log_warn "monitor-web   RUNNING  (PID ${monitor_pid}, not responding yet)"
            fi
        else
            log_warn "monitor-web   STOPPED  (stale PID ${monitor_pid})"
        fi
    else
        log_warn "monitor-web   STOPPED"
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
        monitor|monitor-web)
            local log_file="${LOG_DIR}/monitor-web.log"
            [[ -f "${log_file}" ]] || die "No monitor-web log at ${log_file}"
            log "Tailing ${log_file} (Ctrl+C to stop)"
            tail -f "${log_file}"
            ;;
        *) die "Unknown log target '${target}'. Use: rpc | llama | monitor" ;;
    esac
}

# ------------------------------------------------------------------------------
# Usage
# ------------------------------------------------------------------------------
usage() {
    cat <<EOF
${BOLD}rz-rpc-llm${RESET} — LLaMA.cpp deployment (distributed text or vision)

${BOLD}USAGE${RESET}
  ./deploy.sh <command> [options]

${BOLD}COMMANDS${RESET}
  ${CYAN}clone${RESET}    [--tag TAG]
       Clone llama.cpp at HEAD, or use --tag TAG / --tag latest

  ${CYAN}build-dgx${RESET}
       Rsync source to DGX and build with CUDA (SM121)

  ${CYAN}build-mac${RESET}
       Build locally on Mac with Metal

  ${CYAN}download${RESET} [--repo REPO] [--pattern GLOB] [--dir DIR] [--vision]
       Download model from HuggingFace
       ${GREEN}--vision${RESET} downloads Qwen3.5-122B model + mmproj for multimodal

  ${CYAN}start-rpc${RESET}
       Start rpc-server on DGX (for distributed text inference)

  ${CYAN}stop-rpc${RESET}
       Stop rpc-server on DGX

  ${CYAN}start-llama${RESET} [--model-file PATH] [--alias NAME] [--ctx N] [--parallel N] [--vision]
       Start llama-server locally
       ${GREEN}--vision${RESET} uses vision defaults (no RPC, includes --mmproj)
       PATH is relative to MODELS_DIR or absolute

  ${CYAN}stop-llama${RESET}
       Stop local llama-server

  ${CYAN}start-monitor-web${RESET} [--port PORT]
       Start a colorful browser dashboard that shares the same public port as the API

  ${CYAN}stop-monitor-web${RESET}
       Stop the API + monitor gateway

  ${CYAN}deploy${RESET}   [--tag TAG] [--model-file PATH] [--alias NAME]
                 [--skip-clone] [--skip-build] [--skip-download]
       Full pipeline: clone → build-dgx → build-mac → download → start-rpc → start-llama

  ${CYAN}full${RESET}     [--tag TAG] [--model-file PATH] [--alias NAME]
                 [--skip-clone] [--skip-build] [--skip-download]
       Shortcut for ${CYAN}deploy${RESET}

  ${CYAN}monitor${RESET}  [INTERVAL_SEC] [TABLE_WIDTH]
       Heartbeat rolling table — default 30s interval, 180-char table width
       Ctrl+C gracefully stops all servers

  ${CYAN}status${RESET}
       Show running process status for both devices

  ${CYAN}logs${RESET}     [llama|rpc|monitor]
       Tail logs (default: llama)

${BOLD}EXAMPLES${RESET}
  ${BOLD}# Distributed text inference (Mac + DGX):${RESET}
  ./deploy.sh deploy --tag b8223

  ${BOLD}# Vision model (Mac solo, no DGX needed):${RESET}
  ./deploy.sh download --vision
  ./deploy.sh start-llama --vision

  # Restart vision model
  ./deploy.sh stop-llama
  ./deploy.sh start-llama --vision

  ${BOLD}# Check health${RESET}
  ./deploy.sh status
  curl http://localhost:${LLAMA_PORT:-8680}/health

  ${BOLD}# Start the web monitor${RESET}
  ./deploy.sh start-monitor-web
  open http://127.0.0.1:${MONITOR_WEB_PORT:-8680}/monitor

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
        start-monitor-web) cmd_start_monitor_web "$@" ;;
        stop-monitor-web)  cmd_stop_monitor_web "$@" ;;
        deploy|full) cmd_deploy "$@" ;;
        monitor)     cmd_monitor "$@" ;;
        status)      cmd_status "$@" ;;
        logs)        cmd_logs "$@" ;;
        help|--help|-h) usage ;;
        *) log_error "Unknown command: ${cmd}"; usage; exit 1 ;;
    esac
}

main "$@"
