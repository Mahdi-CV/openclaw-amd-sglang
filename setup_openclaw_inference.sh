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
#   --model-preset PRESET        Qwen3.5 preset — sets model path, served name,
#                                and context window automatically. Presets:
#                                  qwen3.5-0.8b  qwen3.5-2b   qwen3.5-4b
#                                  qwen3.5-9b    qwen3.5-27b  qwen3.5-35b-a3b
#                                  qwen3.5-122b  qwen3.5-397b
#   --model MODEL_PATH           HuggingFace model path (overrides preset)
#   --served-name NAME           Name exposed in the API (default: from preset)
#
# Common options:
#   --port PORT                  Server port (default: 8090)
#   --api-key KEY                API key (default: abc-123)
#   --hf-cache PATH              HuggingFace cache dir (default: /data/hf_cache)
#   --tp-size N                  Tensor parallel size (default: 1)
#   --no-wait                    Don't wait for server health before OpenClaw
#   --server-only                Only start the inference server, skip OpenClaw
#   --openclaw-only              Only install OpenClaw (server must be running)
# =============================================================================

set -euo pipefail

# ---- Defaults ---------------------------------------------------------------
ENGINE="sglang"
MODEL_PRESET=""
MODEL=""
SERVED_NAME=""
CONTEXT_WINDOW=131072
PORT=8090
API_KEY="abc-123"
HF_CACHE="/data/hf_cache"
TP_SIZE=1
WAIT_FOR_SERVER=true
RUN_SERVER=true
RUN_OPENCLAW=true

SGLANG_IMAGE="lmsysorg/sglang:v0.5.9-rocm700-mi30x"
VLLM_IMAGE="vllm/vllm-openai-rocm:v0.15.0"
SERVER_TIMEOUT=3600  # 1 hour — accounts for Docker pull + model download

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

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- Validate engine --------------------------------------------------------
if [[ "$ENGINE" != "sglang" && "$ENGINE" != "vllm" ]]; then
    echo "ERROR: --engine must be 'sglang' or 'vllm', got: $ENGINE"
    exit 1
fi

# ---- Resolve model preset ---------------------------------------------------
if [[ -n "$MODEL_PRESET" ]]; then
    case "$MODEL_PRESET" in
        qwen3.5-0.8b)
            MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
            SERVED_NAME="${SERVED_NAME:-qwen3.5-0.8b}"
            CONTEXT_WINDOW=262144
            ;;
        qwen3.5-2b)
            MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
            SERVED_NAME="${SERVED_NAME:-qwen3.5-2b}"
            CONTEXT_WINDOW=262144
            ;;
        qwen3.5-4b)
            MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
            SERVED_NAME="${SERVED_NAME:-qwen3.5-4b}"
            CONTEXT_WINDOW=262144
            ;;
        qwen3.5-9b)
            MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
            SERVED_NAME="${SERVED_NAME:-qwen3.5-9b}"
            CONTEXT_WINDOW=262144
            ;;
        qwen3.5-27b)
            MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
            SERVED_NAME="${SERVED_NAME:-qwen3.5-27b}"
            CONTEXT_WINDOW=262144
            ;;
        qwen3.5-35b-a3b)
            MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
            SERVED_NAME="${SERVED_NAME:-qwen3.5-35b-a3b}"
            CONTEXT_WINDOW=262144
            ;;
        qwen3.5-122b)
            MODEL="${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
            SERVED_NAME="${SERVED_NAME:-qwen3.5-122b}"
            CONTEXT_WINDOW=131072
            ;;
        qwen3.5-397b)
            MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
            SERVED_NAME="${SERVED_NAME:-qwen3.5-397b}"
            CONTEXT_WINDOW=131072
            ;;
        *)
            echo "ERROR: Unknown --model-preset '$MODEL_PRESET'."
            echo "Valid presets: qwen3.5-0.8b, qwen3.5-2b, qwen3.5-4b, qwen3.5-9b,"
            echo "               qwen3.5-27b, qwen3.5-35b-a3b, qwen3.5-122b, qwen3.5-397b"
            exit 1
            ;;
    esac
fi

# Require a model to be set
if [[ -z "$MODEL" ]]; then
    echo "ERROR: Specify --model-preset PRESET or --model MODEL_PATH."
    exit 1
fi

# Derive served name from model path if still unset
if [[ -z "$SERVED_NAME" ]]; then
    SERVED_NAME=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
fi

CONTAINER_NAME="${ENGINE}_server"

# ---- Detect public IP -------------------------------------------------------
PUBLIC_IP=$(curl -sf --max-time 5 ifconfig.me || curl -sf --max-time 5 api.ipify.org || hostname -I | awk '{print $1}')
BASE_URL="http://${PUBLIC_IP}:${PORT}/v1"

# =============================================================================
# STEP 1 — Launch inference server
# =============================================================================
if $RUN_SERVER; then
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
        deadline=$(( $(date +%s) + SERVER_TIMEOUT ))
        last_log=$(date +%s)
        while [[ $(date +%s) -lt $deadline ]]; do
            if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
                log "  $ENGINE is ready! ($(( $(date +%s) - (deadline - SERVER_TIMEOUT) ))s elapsed)"
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
        log "  Skipping wait (--no-wait). OpenClaw install will proceed immediately."
    fi
fi

# =============================================================================
# STEP 2 — Install & configure OpenClaw
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
        apt-get remove -y libnode-dev nodejs-doc 2>/dev/null || true
        apt-get install -y nodejs
    else
        log "  Node $(node --version) detected — OK."
    fi

    log "  Installing openclaw npm package..."
    SHARP_IGNORE_GLOBAL_LIBVIPS=1 npm install -g openclaw@latest
    log "  OpenClaw installed: $(openclaw --version 2>/dev/null || echo 'version unknown')"

    log ""
    log "  Configuring OpenClaw:"
    log "    Engine : $ENGINE"
    log "    URL    : $BASE_URL"
    log "    Model  : $SERVED_NAME"
    log ""

    openclaw config set gateway.mode local || true
    openclaw config set agents.defaults.model "${ENGINE}/${SERVED_NAME}" || true

    CONFIG_FILE="${OPENCLAW_CONFIG_PATH:-$HOME/.openclaw/openclaw.json}"
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
    GATEWAY_PID=$!
    sleep 3
    if kill -0 $GATEWAY_PID 2>/dev/null; then
        log "  Gateway running (PID $GATEWAY_PID). Logs: tail -f /tmp/openclaw-gateway.log"
    else
        log "  Gateway may have failed — check: tail -f /tmp/openclaw-gateway.log"
    fi

    DASHBOARD_URL=$(openclaw dashboard --no-open 2>/dev/null | grep -o 'http://[^ ]*' | head -1 || true)
fi

# =============================================================================
# Summary
# =============================================================================
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
    log "  1. Run this SSH tunnel on your local machine:"
    log "     ssh -N -L 18789:127.0.0.1:18789 amd@$PUBLIC_IP"
    log ""
    if [[ -n "${DASHBOARD_URL:-}" ]]; then
        log "  2. Open this URL (token included):"
        log "     $DASHBOARD_URL"
    else
        log "  2. Open: http://localhost:18789"
        log "     (run 'openclaw dashboard' on the server to get the token URL)"
    fi
    log ""
    log "  Gateway logs  : tail -f /tmp/openclaw-gateway.log"
    log "  Status        : openclaw status"
fi
log "============================================================"
