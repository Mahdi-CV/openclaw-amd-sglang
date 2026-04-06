#!/bin/bash
# =============================================================================
# setup_openclaw_inference.sh
#
# Launches an OpenAI-compatible inference server (SGLang or vLLM) and installs
# OpenClaw configured to use it. Supports the full Qwen3.5 model family.
#
# Usage:
#   bash setup_openclaw_inference.sh [options]
#
# Engine:
#   --engine sglang|vllm         Inference engine (default: sglang)
#
# Model (pick one):
#   --model-preset PRESET        Qwen3.5 preset. Run without args to see list.
#   --model MODEL_PATH           HuggingFace model path (custom)
#   --served-name NAME           Name exposed in the API (default: from preset)
#
# Common options:
#   --port PORT                  Server port (default: 8090)
#   --api-key KEY                API key (default: abc-123)
#   --hf-cache PATH              HuggingFace cache dir
#                                (default: $HOME/.cache/huggingface)
#   --tp-size N                  Tensor parallel size (default: 1)
#   --no-wait                    Don't wait for server health before OpenClaw
#   --server-only                Only start the inference server, skip OpenClaw
#   --openclaw-only              Only install OpenClaw (server must be running)
# =============================================================================

set -euo pipefail

# ---- Helpers ----------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }
die()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

print_banner() {
    printf '\033[1;31m░████░░█░░░█░█████░░████░█░░░█░░░░░████░░▀█▀░░████░░████░░▀█▀\033[0m\n'
    printf '\033[1;31m█░░░█░░█░░░█░░░█░░░█░░░░░█░██░░░░░█░░░░░░█░░░█░░░█░█░░░█░░█░\033[0m\n'
    printf '\033[1;31m█░░░█░░█░░░█░░░█░░░█░░░░░██░░░░░░░░███░░░█░░░█████░████░░░█░\033[0m\n'
    printf '\033[1;31m█░█░█░░█░░░█░░░█░░░█░░░░░█░██░░░░░░░░░█░░█░░░█░░░█░█░░█░░░█░\033[0m\n'
    printf '\033[1;31m░█░█░░░░███░░░█░░░░████░█░░░█░░░░░████░░░█░░░█░░░█░█░░░█░░█░\033[0m\n'
    printf '\033[1;33m    🦞  OpenClaw on AMD (SGLang / vLLM)  🦞\033[0m\n'
    printf '\n'
}

# ---- Defaults ---------------------------------------------------------------
ENGINE="sglang"
MODEL_PRESET="qwen3.5-122b"
MODEL=""
SERVED_NAME=""
CONTEXT_WINDOW=131072
PORT=8090
API_KEY="abc-123"
HF_CACHE="${HOME}/.cache/huggingface"
TP_SIZE=1
WAIT_FOR_SERVER=true
RUN_SERVER=true
RUN_OPENCLAW=true

# Detect interactive mode once at startup.
# true  = stdin is a real terminal → show prompts
# false = piped / SSH no-PTY / CI  → auto-accept and use defaults
[[ -t 0 ]] && INTERACTIVE=true || INTERACTIVE=false

SGLANG_IMAGE="lmsysorg/sglang:v0.5.9-rocm700-mi30x"
VLLM_IMAGE="vllm/vllm-openai-rocm:v0.15.0"
SERVER_TIMEOUT=3600  # 1 hour — accounts for Docker pull + model download

# ---- Preset table -----------------------------------------------------------
# Parallel arrays — index-matched. Keep in sync.
PRESET_KEYS=(
    "qwen3.5-0.8b"
    "qwen3.5-2b"
    "qwen3.5-4b"
    "qwen3.5-9b"
    "qwen3.5-27b"
    "qwen3.5-35b-a3b"
    "qwen3.5-122b"
    "qwen3.5-397b"
)
PRESET_MODELS=(
    "Qwen/Qwen3.5-0.8B"
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-35B-A3B"
    "Qwen/Qwen3.5-122B-A10B-FP8"
    "Qwen/Qwen3.5-397B-A17B-FP8"
)
PRESET_CTX=(262144 262144 262144 262144 262144 262144 131072 131072)
PRESET_GB=(2 5 9 20 55 20 65 200)

# Set by resolve_preset, used by check_disk_space
PRESET_NEEDED_GB=0

# Set in main after preset resolution
CONTAINER_NAME=""
PUBLIC_IP=""
BASE_URL=""

# ---- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine)        ENGINE="$2";        shift 2 ;;
        --model-preset)  MODEL_PRESET="$2";  shift 2 ;;
        --model)         MODEL="$2";         shift 2 ;;
        --served-name)   SERVED_NAME="$2";   shift 2 ;;
        --port)          PORT="$2";          shift 2 ;;
        --api-key)       API_KEY="$2";       shift 2 ;;
        --hf-cache)      HF_CACHE="$2";      shift 2 ;;
        --tp-size)       TP_SIZE="$2";       shift 2 ;;
        --no-wait)       WAIT_FOR_SERVER=false; shift ;;
        --server-only)   RUN_OPENCLAW=false; shift ;;
        --openclaw-only) RUN_SERVER=false; WAIT_FOR_SERVER=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Validate engine --------------------------------------------------------
if [[ "$ENGINE" != "sglang" && "$ENGINE" != "vllm" ]]; then
    echo "ERROR: --engine must be 'sglang' or 'vllm', got: $ENGINE"
    exit 1
fi

# =============================================================================
# Risk acknowledgement
# =============================================================================
risk_acknowledgement() {
    printf '\n'
    printf '\033[1;33m=================================================================\033[0m\n'
    printf '\033[1;33m  IMPORTANT — PLEASE READ BEFORE CONTINUING\033[0m\n'
    printf '\033[1;33m=================================================================\033[0m\n'
    printf '\n'
    printf '\033[1;33mOpenClaw is a highly autonomous AI agent. Giving any AI agent\033[0m\n'
    printf '\033[1;33maccess to any system may result in the AI acting in unpredictable\033[0m\n'
    printf '\033[1;33mways with unpredictable/unforeseen outcomes. Use of any AMD\033[0m\n'
    printf '\033[1;33msuggested implementations is made at your own risk. AMD makes no\033[0m\n'
    printf '\033[1;33mrepresentations/warranties with your use of an AI agent as\033[0m\n'
    printf '\033[1;33mdescribed herein. Failure to exercise appropriate caution may\033[0m\n'
    printf '\033[1;33mresult in damages (foreseen and/or unforeseen).\033[0m\n'
    printf '\n'
    printf '\033[1;33m=================================================================\033[0m\n'
    printf '\n'
    if ! $INTERACTIVE; then
        log "Non-interactive session — risk auto-accepted."
        return 0
    fi
    printf 'Do you accept and wish to continue? [y/N]: '
    local accept=""
    read -r accept
    [[ "$accept" =~ ^[Yy] ]] || die "Risk not accepted. Exiting."
    printf '\n'
}

# =============================================================================
# Prerequisite checks
# =============================================================================
check_docker() {
    have docker || {
        log "ERROR: Docker is not installed."
        log "       Install it: https://docs.docker.com/engine/install/"
        exit 1
    }
    docker info >/dev/null 2>&1 || {
        log "ERROR: Docker daemon is not running."
        log "       Start it: sudo systemctl start docker"
        exit 1
    }
    log "  Docker      : OK ($(docker --version | awk '{print $3}' | tr -d ','))"
}

check_rocm() {
    local missing=()

    # Check kernel-level ROCm devices
    [[ -e /dev/kfd ]] || missing+=("/dev/kfd")
    [[ -d /dev/dri ]] || missing+=("/dev/dri")

    # Check for ROCm userspace tools
    have rocminfo || have rocm-smi || have amd-smi || missing+=("rocm userspace tools")

    if [[ ${#missing[@]} -gt 0 ]]; then
        printf '\n'
        printf '\033[1;31m=================================================================\033[0m\n'
        printf '\033[1;31m  ROCm NOT DETECTED — Cannot continue\033[0m\n'
        printf '\033[1;31m=================================================================\033[0m\n'
        printf '\n'
        printf '\033[1;31mThe following ROCm components were not found:\033[0m\n'
        for item in "${missing[@]}"; do
            printf '\033[1;31m  - %s\033[0m\n' "$item"
        done
        printf '\n'
        printf '\033[1;33mROCm must be installed on the host before running this script.\033[0m\n'
        printf '\033[1;33mThis script does not install ROCm — it only uses it.\033[0m\n'
        printf '\n'
        printf '\033[1;33mPlease follow the AMD ROCm quick-start installation guide:\033[0m\n'
        printf '\033[1;36m  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html\033[0m\n'
        printf '\n'
        printf '\033[1;33mAfter installing ROCm, re-run this script.\033[0m\n'
        printf '\033[1;31m=================================================================\033[0m\n'
        printf '\n'
        exit 1
    fi

    log "  ROCm        : OK (/dev/kfd, /dev/dri present)"
}

check_gpu() {
    if have rocm-smi; then
        local gpu_count
        gpu_count=$(rocm-smi --showid 2>/dev/null | grep -c 'GPU\[' || echo "?")
        log "  GPU         : ${gpu_count} AMD GPU(s) detected via rocm-smi"
    elif have amd-smi; then
        log "  GPU         : AMD GPU detected via amd-smi"
    else
        log "  GPU         : WARNING — rocm-smi/amd-smi not found, cannot verify GPU"
    fi
}

# =============================================================================
# HF cache helpers
# =============================================================================

# Check if a model (by HF path e.g. "Qwen/Qwen3.5-9B") is already downloaded.
# A model is considered cached if its blobs directory exists and is non-empty.
model_is_cached() {
    local model_path="$1"
    local org="${model_path%%/*}"
    local name="${model_path#*/}"
    local hub_dir="${HF_CACHE}/hub/models--${org}--${name}"
    [[ -d "${hub_dir}/blobs" ]] && [[ -n "$(ls -A "${hub_dir}/blobs" 2>/dev/null)" ]]
}

check_disk_space() {
    local model_path="$1"
    local needed_gb="$2"

    # If already cached, no download needed
    if model_is_cached "$model_path"; then
        log "  Disk        : model already in cache — no download needed"
        return 0
    fi

    # Custom model with no size estimate
    if [[ "$needed_gb" -eq 0 ]]; then
        log "  Disk        : custom model — cannot estimate download size"
        return 0
    fi

    mkdir -p "$HF_CACHE" 2>/dev/null || true
    local avail_kb avail_gb
    avail_kb=$(df -k "$HF_CACHE" 2>/dev/null | awk 'NR==2{print $4}' \
               || df -k / | awk 'NR==2{print $4}')
    avail_gb=$(( avail_kb / 1048576 ))

    log "  Disk        : ${avail_gb}GB available, ~${needed_gb}GB needed for download"
    if [[ $avail_gb -lt $needed_gb ]]; then
        log "  WARNING: Low disk space — download may fail."
        log "           Use --hf-cache PATH to point to a volume with more space."
    fi
}

# =============================================================================
# Preset resolution + interactive picker
# =============================================================================

resolve_preset() {
    local i
    for i in "${!PRESET_KEYS[@]}"; do
        if [[ "${PRESET_KEYS[$i]}" == "$MODEL_PRESET" ]]; then
            MODEL="${MODEL:-${PRESET_MODELS[$i]}}"
            SERVED_NAME="${SERVED_NAME:-${PRESET_KEYS[$i]}}"
            CONTEXT_WINDOW="${PRESET_CTX[$i]}"
            PRESET_NEEDED_GB="${PRESET_GB[$i]}"
            return 0
        fi
    done
    echo "ERROR: Unknown --model-preset '$MODEL_PRESET'."
    printf "Valid presets: %s\n" "${PRESET_KEYS[*]}"
    exit 1
}

# Called when neither --model-preset nor --model is given.
# Scans the HF cache, marks already-downloaded models, and lets the user pick.
pick_preset() {
    # Non-interactive: no TTY to prompt on — use the built-in default.
    if ! $INTERACTIVE; then
        log "Non-interactive session — using default preset: $MODEL_PRESET"
        return 0
    fi

    log "No --model-preset specified."
    log "Scanning HF cache at: $HF_CACHE"
    printf '\n'
    printf '  %-4s %-20s %-32s %6s  %s\n' "No." "Preset" "HuggingFace model" "Size" "Cache"
    printf '  %-4s %-20s %-32s %6s  %s\n' "---" "------" "-----------------" "----" "-----"

    local i
    for i in "${!PRESET_KEYS[@]}"; do
        local cached_tag=""
        model_is_cached "${PRESET_MODELS[$i]}" && cached_tag="\033[0;32m[cached]\033[0m"
        printf "  [%d]  %-20s %-32s ~%3dGB  %b\n" \
            "$(( i + 1 ))" \
            "${PRESET_KEYS[$i]}" \
            "${PRESET_MODELS[$i]}" \
            "${PRESET_GB[$i]}" \
            "$cached_tag"
    done

    printf '\n'
    printf 'Select [1-%d]: ' "${#PRESET_KEYS[@]}"
    local choice
    read -r choice
    printf '\n'

    if [[ "$choice" =~ ^[0-9]+$ ]] && \
       [[ "$choice" -ge 1 ]] && \
       [[ "$choice" -le "${#PRESET_KEYS[@]}" ]]; then
        MODEL_PRESET="${PRESET_KEYS[$((choice - 1))]}"
        log "Selected: $MODEL_PRESET"
    else
        log "ERROR: Invalid selection. Use --model-preset PRESET to specify directly."
        exit 1
    fi
}

# =============================================================================
# STEP 1 — Launch inference server
# =============================================================================
launch_server() {
    log "============================================================"
    log " STEP 1: Launching $ENGINE server"
    log "============================================================"
    log "  Engine     : $ENGINE"
    log "  Model      : $MODEL"
    log "  Served as  : $SERVED_NAME"
    log "  Port       : $PORT"
    log "  TP size    : $TP_SIZE"
    log "  HF cache   : $HF_CACHE"
    log "  Container  : $CONTAINER_NAME"

    if model_is_cached "$MODEL"; then
        log "  Cache      : model found — starting without download"
    else
        log "  Cache      : model not cached — will download on first run"
    fi

    docker rm -f "$CONTAINER_NAME" 2>/dev/null && log "  Removed existing container." || true
    mkdir -p "$HF_CACHE"

    if [[ "$ENGINE" == "sglang" ]]; then
        docker run -d \
          --name "$CONTAINER_NAME" \
          --device=/dev/kfd --device=/dev/dri \
          --ipc=host --shm-size 32G \
          --group-add video --cap-add=SYS_PTRACE \
          --security-opt seccomp=unconfined \
          -p "${PORT}:${PORT}" \
          -v "${HF_CACHE}:/root/.cache/huggingface" \
          "$SGLANG_IMAGE" \
          python3 -m sglang.launch_server \
            --model-path "$MODEL" \
            --served-model-name "$SERVED_NAME" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --tp-size "$TP_SIZE" \
            --api-key "$API_KEY" \
            --mem-fraction-static 0.85 \
            --attention-backend triton \
            --reasoning-parser qwen3 \
            --tool-call-parser qwen3_coder \
            --trust-remote-code

    elif [[ "$ENGINE" == "vllm" ]]; then
        docker run -d \
          --name "$CONTAINER_NAME" \
          --device=/dev/kfd --device=/dev/dri \
          --ipc=host --shm-size 32G \
          --group-add video --cap-add=SYS_PTRACE \
          --security-opt seccomp=unconfined \
          -p "${PORT}:${PORT}" \
          -v "${HF_CACHE}:/root/.cache/huggingface" \
          -e VLLM_ROCM_USE_AITER=1 \
          "$VLLM_IMAGE" \
          --model "$MODEL" \
          --served-model-name "$SERVED_NAME" \
          --host 0.0.0.0 \
          --port "$PORT" \
          --tensor-parallel-size "$TP_SIZE" \
          --api-key "$API_KEY" \
          --reasoning-parser qwen3 \
          --enable-auto-tool-choice \
          --tool-call-parser hermes \
          --trust-remote-code \
          --enable-prefix-caching
    fi

    log "  Container started. ID: $(docker ps -q --filter name="$CONTAINER_NAME")"

    if $WAIT_FOR_SERVER; then
        log "  Waiting for $ENGINE health endpoint (up to ${SERVER_TIMEOUT}s)..."
        local deadline last_log now
        deadline=$(( $(date +%s) + SERVER_TIMEOUT ))
        last_log=$(date +%s)
        while [[ $(date +%s) -lt $deadline ]]; do
            if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
                log "  $ENGINE is ready! ($(( $(date +%s) - (deadline - SERVER_TIMEOUT) ))s elapsed)"
                break
            fi
            local state
            state=$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "gone")
            if [[ "$state" != "running" ]]; then
                log "  ERROR: Container stopped unexpectedly. Last logs:"
                docker logs "$CONTAINER_NAME" --tail 20
                exit 1
            fi
            now=$(date +%s)
            if (( now - last_log >= 60 )); then
                log "  Still loading... $(( now - (deadline - SERVER_TIMEOUT) ))s elapsed"
                last_log=$now
            fi
            sleep 10
        done
        if ! curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
            log "  ERROR: $ENGINE did not become ready within ${SERVER_TIMEOUT}s."
            docker logs "$CONTAINER_NAME" --tail 30
            exit 1
        fi
    else
        log "  Skipping wait (--no-wait)."
    fi
}

# =============================================================================
# STEP 2 — Install & configure OpenClaw
# =============================================================================
install_openclaw() {
    log ""
    log "============================================================"
    log " STEP 2: Installing OpenClaw"
    log "============================================================"

    local NODE_MAJOR
    NODE_MAJOR=$(node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/' || echo "0")
    if [[ "$NODE_MAJOR" -lt 22 ]]; then
        log "  Node $NODE_MAJOR detected — installing Node 22 LTS..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
        apt-get remove -y libnode-dev nodejs-doc 2>/dev/null || true
        apt-get install -y nodejs
    else
        log "  Node $(node --version) — OK"
    fi

    log "  Installing openclaw npm package..."
    SHARP_IGNORE_GLOBAL_LIBVIPS=1 npm install -g openclaw@latest
    log "  OpenClaw: $(openclaw --version 2>/dev/null || echo 'installed')"

    log ""
    log "  Configuring OpenClaw:"
    log "    Engine : $ENGINE"
    log "    URL    : $BASE_URL"
    log "    Model  : $SERVED_NAME"

    openclaw config set gateway.mode local || true
    openclaw config set agents.defaults.model "${ENGINE}/${SERVED_NAME}" || true

    local CONFIG_FILE="${OPENCLAW_CONFIG_PATH:-$HOME/.openclaw/openclaw.json}"
    node - <<JSEOF
const fs = require('fs');
const cfg = JSON.parse(fs.readFileSync('${CONFIG_FILE}', 'utf8'));
cfg.models = cfg.models || {};
cfg.models.providers = cfg.models.providers || {};
cfg.models.providers['${ENGINE}'] = {
  baseUrl: "${BASE_URL}",
  apiKey: "${API_KEY}",
  api: "openai-completions",
  models: [{
    id: "${SERVED_NAME}",
    name: "${SERVED_NAME}",
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: ${CONTEXT_WINDOW},
    maxTokens: 8192
  }]
};
fs.writeFileSync('${CONFIG_FILE}', JSON.stringify(cfg, null, 2));
console.log('  models.providers.${ENGINE} configured.');
JSEOF

    chmod 700 "$HOME/.openclaw" 2>/dev/null || true
    chmod 600 "$HOME/.openclaw/openclaw.json" 2>/dev/null || true
    mkdir -p "$HOME/.openclaw/agents/main/sessions"

    openclaw doctor --fix 2>/dev/null || true
    loginctl enable-linger "$(whoami)" 2>/dev/null || true

    log "  Installing OpenClaw gateway service..."
    openclaw gateway install 2>/dev/null || true

    log "  Starting OpenClaw gateway..."
    pkill -9 -f openclaw-gateway 2>/dev/null || true
    nohup openclaw gateway run --bind loopback --port 18789 --force \
      > /tmp/openclaw-gateway.log 2>&1 &
    local GATEWAY_PID=$!
    sleep 3
    if kill -0 $GATEWAY_PID 2>/dev/null; then
        log "  Gateway running (PID $GATEWAY_PID)"
    else
        log "  Gateway may have failed — check: tail -f /tmp/openclaw-gateway.log"
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    print_banner
    risk_acknowledgement

    # Prerequisites — only needed when launching a server container
    if $RUN_SERVER; then
        log "============================================================"
        log " Checking prerequisites"
        log "============================================================"
        check_docker
        check_rocm
        check_gpu
    fi

    # Model resolution — interactive picker if nothing specified
    if [[ -z "$MODEL_PRESET" && -z "$MODEL" ]]; then
        pick_preset
    fi
    if [[ -n "$MODEL_PRESET" ]]; then
        resolve_preset
    fi

    [[ -z "$MODEL" ]] && {
        log "ERROR: Specify --model-preset PRESET or --model MODEL_PATH."
        exit 1
    }

    # Derive served name from model path if still unset
    [[ -z "$SERVED_NAME" ]] && \
        SERVED_NAME=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')

    CONTAINER_NAME="${ENGINE}_server"
    PUBLIC_IP=$(curl -sf --max-time 5 ifconfig.me \
                || curl -sf --max-time 5 api.ipify.org \
                || hostname -I | awk '{print $1}')
    BASE_URL="http://${PUBLIC_IP}:${PORT}/v1"

    # Disk space check — runs after model is known, skips if already cached
    if $RUN_SERVER; then
        check_disk_space "$MODEL" "$PRESET_NEEDED_GB"
    fi

    if $RUN_SERVER; then
        launch_server
    fi

    if $RUN_OPENCLAW; then
        install_openclaw
    fi

    # Summary
    local DASHBOARD_URL=""
    if $RUN_OPENCLAW; then
        DASHBOARD_URL=$(openclaw dashboard --no-open 2>/dev/null \
                        | grep -o 'http://[^ ]*' | head -1 || true)
    fi

    log ""
    log "============================================================"
    log " Setup complete!"
    log "============================================================"
    if $RUN_SERVER; then
        log "  Engine        : $ENGINE"
        log "  Server URL    : http://localhost:${PORT}"
        log "  OpenAI URL    : $BASE_URL"
        log "  API key       : $API_KEY"
        log "  Model name    : $SERVED_NAME"
        log ""
        log "  Server logs   : docker logs -f $CONTAINER_NAME"
    fi
    if $RUN_OPENCLAW; then
        log ""
        log "  ---- Connect to the browser UI ----"
        log "  1. SSH tunnel : ssh -N -L 18789:127.0.0.1:18789 amd@$PUBLIC_IP"
        log ""
        if [[ -n "${DASHBOARD_URL:-}" ]]; then
            log "  2. Open: $DASHBOARD_URL"
        else
            log "  2. Open: http://localhost:18789"
            log "     (run 'openclaw dashboard' on the server to get the token URL)"
        fi
        log ""
        log "  Gateway logs  : tail -f /tmp/openclaw-gateway.log"
        log "  Status        : openclaw status"
    fi
    log "============================================================"
}

main "$@"
