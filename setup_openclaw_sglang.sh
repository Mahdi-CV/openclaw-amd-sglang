#!/bin/bash
# =============================================================================
# setup_openclaw_sglang.sh
#
# All-in-one: launches SGLang (Qwen3.5-122B) + installs & configures OpenClaw
# in a single run. No manual steps required.
#
# Usage:
#   bash setup_openclaw_sglang.sh [options]
#
# Options:
#   --port PORT          SGLang port (default: 8090)
#   --api-key KEY        SGLang API key (default: abc-123)
#   --model MODEL_PATH   HuggingFace model path (default: Qwen/Qwen3.5-122B-A10B-FP8)
#   --served-name NAME   Served model name (default: qwen3-5-122b)
#   --hf-cache PATH      HuggingFace cache dir (default: /data/hf_cache)
#   --tp-size N          Tensor parallel size (default: 1)
#   --no-wait            Don't wait for SGLang to be ready before installing OpenClaw
#   --sglang-only        Only start SGLang, skip OpenClaw install
#   --openclaw-only      Only install OpenClaw (SGLang assumed already running)
# =============================================================================

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
PORT=8090
API_KEY="abc-123"
MODEL="Qwen/Qwen3.5-122B-A10B-FP8"
SERVED_NAME="qwen3-5-122b"
HF_CACHE="/data/hf_cache"
TP_SIZE=1
IMAGE="lmsysorg/sglang:v0.5.9-rocm700-mi30x"
CONTAINER_NAME="sglang_server"
WAIT_FOR_SGLANG=true
RUN_SGLANG=true
RUN_OPENCLAW=true
SGLANG_TIMEOUT=3600  # 1 hour (accounts for Docker image pull + model download ~100GB)

# ---- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)         PORT="$2";         shift 2 ;;
        --api-key)      API_KEY="$2";      shift 2 ;;
        --model)        MODEL="$2";        shift 2 ;;
        --served-name)  SERVED_NAME="$2";  shift 2 ;;
        --hf-cache)     HF_CACHE="$2";     shift 2 ;;
        --tp-size)      TP_SIZE="$2";      shift 2 ;;
        --no-wait)      WAIT_FOR_SGLANG=false; shift ;;
        --sglang-only)  RUN_OPENCLAW=false; shift ;;
        --openclaw-only) RUN_SGLANG=false; WAIT_FOR_SGLANG=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---- Prerequisite checks ----------------------------------------------------
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
    log "  Docker : OK ($(docker --version | awk '{print $3}' | tr -d ','))"
}

check_rocm() {
    local missing=()

    [[ -e /dev/kfd ]] || missing+=("/dev/kfd")
    [[ -d /dev/dri ]] || missing+=("/dev/dri")
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

    log "  ROCm   : OK (/dev/kfd, /dev/dri present)"
}

# ---- Detect public IP -------------------------------------------------------
PUBLIC_IP=$(curl -sf --max-time 5 ifconfig.me || curl -sf --max-time 5 api.ipify.org || hostname -I | awk '{print $1}')
BASE_URL="http://${PUBLIC_IP}:${PORT}/v1"

# =============================================================================
# Prerequisites
# =============================================================================
if $RUN_SGLANG; then
    log "============================================================"
    log " Checking prerequisites"
    log "============================================================"
    check_docker
    check_rocm
fi

# =============================================================================
# STEP 1 — Launch SGLang
# =============================================================================
if $RUN_SGLANG; then
    log "============================================================"
    log " STEP 1: Launching SGLang server"
    log "============================================================"
    log "  Model      : $MODEL"
    log "  Served as  : $SERVED_NAME"
    log "  Port       : $PORT"
    log "  TP size    : $TP_SIZE"
    log "  HF cache   : $HF_CACHE"
    log "  Container  : $CONTAINER_NAME"

    # Remove any existing container with the same name
    docker rm -f "$CONTAINER_NAME" 2>/dev/null && log "  Removed existing container."

    mkdir -p "$HF_CACHE"

    docker run -d \
      --name "$CONTAINER_NAME" \
      --device=/dev/kfd --device=/dev/dri \
      --ipc=host --shm-size 32G \
      --group-add video --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      -p "${PORT}:${PORT}" \
      -v "${HF_CACHE}:/root/.cache/huggingface" \
      "$IMAGE" \
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

    log "  Container started. ID: $(docker ps -q --filter name=$CONTAINER_NAME)"

    if $WAIT_FOR_SGLANG; then
        log "  Waiting for SGLang health endpoint (up to ${SGLANG_TIMEOUT}s, includes model download)..."
        deadline=$(( $(date +%s) + SGLANG_TIMEOUT ))
        last_log=$(date +%s)
        while [[ $(date +%s) -lt $deadline ]]; do
            if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
                log "  SGLang is ready! ($(( $(date +%s) - (deadline - SGLANG_TIMEOUT) ))s elapsed)"
                break
            fi
            state=$(docker inspect --format '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "gone")
            if [[ "$state" != "running" ]]; then
                log "  ERROR: Container stopped unexpectedly. Last logs:"
                docker logs "$CONTAINER_NAME" --tail 20
                exit 1
            fi
            now=$(date +%s)
            if (( now - last_log >= 60 )); then
                log "  Still loading... $(( now - (deadline - SGLANG_TIMEOUT) ))s elapsed"
                last_log=$now
            fi
            sleep 10
        done
        if ! curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
            log "  ERROR: SGLang did not become ready within ${SGLANG_TIMEOUT}s."
            docker logs "$CONTAINER_NAME" --tail 30
            exit 1
        fi
    else
        log "  Skipping wait (--no-wait). OpenClaw install will proceed immediately."
    fi
fi

# =============================================================================
# STEP 2 — Install OpenClaw and auto-configure it non-interactively
# =============================================================================
if $RUN_OPENCLAW; then
    log ""
    log "============================================================"
    log " STEP 2: Installing OpenClaw"
    log "============================================================"

    # Install Node 22+ if missing or too old
    NODE_MAJOR=$(node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/' || echo "0")
    if [[ "$NODE_MAJOR" -lt 22 ]]; then
        log "  Node $NODE_MAJOR detected — installing Node 22 LTS..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
        apt-get install -y nodejs
    else
        log "  Node $(node --version) detected — OK."
    fi

    log "  Installing openclaw npm package..."
    SHARP_IGNORE_GLOBAL_LIBVIPS=1 npm install -g openclaw@latest

    log "  OpenClaw installed: $(openclaw --version 2>/dev/null || echo 'version unknown')"

    # ---- Non-interactive onboarding via environment variables ---------------
    log ""
    log "  Configuring OpenClaw to connect to SGLang at:"
    log "    URL  : $BASE_URL"
    log "    Model: $SERVED_NAME"
    log "    Key  : $API_KEY"
    log ""

    # ---- Configure via openclaw config set + direct JSON for models provider --
    log "  Configuring OpenClaw..."
    openclaw config set gateway.mode local || true
    # agents.defaults.model uses "provider/model-id" format
    openclaw config set agents.defaults.model "sglang/${SERVED_NAME}" || true

    # models.providers requires a nested structure with a models[] array —
    # config set can't build arrays, so write it directly into the JSON
    CONFIG_FILE="${OPENCLAW_CONFIG_PATH:-$HOME/.openclaw/openclaw.json}"
    node - <<JSEOF
const fs = require('fs');
const cfg = JSON.parse(fs.readFileSync('${CONFIG_FILE}', 'utf8'));
cfg.models = cfg.models || {};
cfg.models.providers = cfg.models.providers || {};
cfg.models.providers.sglang = {
  baseUrl: "${BASE_URL}",
  apiKey: "${API_KEY}",
  api: "openai-completions",
  models: [{ id: "${SERVED_NAME}", name: "${SERVED_NAME}", reasoning: true, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 131072, maxTokens: 8192 }]
};
fs.writeFileSync('${CONFIG_FILE}', JSON.stringify(cfg, null, 2));
console.log('  models.providers.sglang configured.');
JSEOF

    # Fix state directory permissions
    chmod 700 "$HOME/.openclaw" 2>/dev/null || true
    chmod 600 "$HOME/.openclaw/openclaw.json" 2>/dev/null || true

    # Create required session store dir
    mkdir -p "$HOME/.openclaw/agents/main/sessions"

    # Fix any remaining doctor issues
    log "  Running openclaw doctor --fix..."
    openclaw doctor --fix 2>/dev/null || true

    # Enable linger so the systemd user service survives SSH logout/reboot
    loginctl enable-linger "$(whoami)" 2>/dev/null || true

    # Install and start the gateway service
    log "  Installing OpenClaw gateway service..."
    openclaw gateway install 2>/dev/null || true

    log "  Starting OpenClaw gateway..."
    pkill -9 -f openclaw-gateway 2>/dev/null || true
    nohup openclaw gateway run --bind loopback --port 18789 --force \
      > /tmp/openclaw-gateway.log 2>&1 &
    GATEWAY_PID=$!
    sleep 3
    if kill -0 $GATEWAY_PID 2>/dev/null; then
        log "  Gateway running (PID $GATEWAY_PID). Logs: tail -f /tmp/openclaw-gateway.log"
    else
        log "  Gateway may have failed — check: tail -f /tmp/openclaw-gateway.log"
    fi

    # Grab the dashboard URL (includes the auth token in the fragment)
    DASHBOARD_URL=$(openclaw dashboard --no-open 2>/dev/null | grep -o 'http://[^ ]*' | head -1)
    # Rewrite localhost to the public IP for the SSH tunnel hint
    DASHBOARD_URL_REMOTE="${DASHBOARD_URL/localhost/$PUBLIC_IP}"
fi

# =============================================================================
# Summary
# =============================================================================
log ""
log "============================================================"
log " Setup complete!"
log "============================================================"
if $RUN_SGLANG; then
    log "  SGLang server : http://localhost:${PORT}"
    log "  OpenAI URL    : $BASE_URL"
    log "  API key       : $API_KEY"
    log "  Model name    : $SERVED_NAME"
    log ""
    log "  Tail SGLang logs : docker logs -f $CONTAINER_NAME"
fi
if $RUN_OPENCLAW; then
    log ""
    log "  ---- Connect to the browser UI ----"
    log "  1. Run this SSH tunnel on your local machine:"
    log "     ssh -N -L 18789:127.0.0.1:18789 root@$PUBLIC_IP"
    log ""
    if [[ -n "${DASHBOARD_URL:-}" ]]; then
        log "  2. Open this URL (token included):"
        log "     $DASHBOARD_URL"
    else
        log "  2. Open: http://localhost:18789"
        log "     (run 'openclaw dashboard' on the server to get the token URL)"
    fi
    log ""
    log "  Gateway logs : tail -f /tmp/openclaw-gateway.log"
    log "  Status       : openclaw status"
fi
log "============================================================"
