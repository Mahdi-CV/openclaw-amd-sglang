#!/bin/bash
# =============================================================================
# run_all.sh — Benchmark all Qwen3.5 presets × SGLang + vLLM
#
# Iterates every engine × model combination sequentially, launches the
# inference container, waits for the health endpoint, runs bench.py, saves
# the result as benchmarks/results/{engine}-{preset}.json, then tears down.
#
# Usage:
#   bash benchmarks/run_all.sh [--hf-cache PATH] [--only-engine sglang|vllm]
#                              [--only-preset PRESET] [--skip-format]
#
# Resume: already-completed runs are skipped (result file exists).
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

# ---- Config -----------------------------------------------------------------
PORT=8090
API_KEY="abc-123"
HF_CACHE="${HOME}/.cache/huggingface"
HEALTH_TIMEOUT=7200   # 2 hours: covers Docker pull + model download + load
COOLDOWN=30           # seconds to wait after container stop for VRAM to clear
REQUESTS=16           # requests per concurrency level
CONCURRENCY="1 4 16"

SGLANG_IMAGE="lmsysorg/sglang:v0.5.9-rocm700-mi30x"
VLLM_IMAGE="vllm/vllm-openai-rocm:v0.15.0"

# ---- Preset table (keep in sync with setup_openclaw_inference.sh) ----------
PRESET_KEYS=(
    qwen3.5-0.8b
    qwen3.5-2b
    qwen3.5-4b
    qwen3.5-9b
    qwen3.5-27b
    qwen3.5-35b-a3b
    qwen3.5-122b
    qwen3.5-397b
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

# Tensor-parallel size per preset
declare -A PRESET_TP=(
    [qwen3.5-0.8b]=1
    [qwen3.5-2b]=1
    [qwen3.5-4b]=1
    [qwen3.5-9b]=1
    [qwen3.5-27b]=1
    [qwen3.5-35b-a3b]=1
    [qwen3.5-122b]=1
    [qwen3.5-397b]=2
)

# ROCR_VISIBLE_DEVICES override — use GPUs 1,2 for 397B to avoid GPU 0
# (GPU 0 may be occupied by other workloads on the server)
declare -A PRESET_GPU_VIS=(
    [qwen3.5-397b]="1,2"
)

# ---- CLI args ---------------------------------------------------------------
ONLY_ENGINE=""
ONLY_PRESET=""
SKIP_FORMAT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf-cache)     HF_CACHE="$2"; shift 2 ;;
        --only-engine)  ONLY_ENGINE="$2"; shift 2 ;;
        --only-preset)  ONLY_PRESET="$2"; shift 2 ;;
        --skip-format)  SKIP_FORMAT=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Helpers ----------------------------------------------------------------
RED='\033[1;31m'; GRN='\033[0;32m'; YLW='\033[1;33m'; CYN='\033[1;36m'; RST='\033[0m'
log()  { printf "${CYN}[%s]${RST} %s\n" "$(date '+%H:%M:%S')" "$*"; }
ok()   { printf "${GRN}[OK]${RST} %s\n" "$*"; }
warn() { printf "${YLW}[WARN]${RST} %s\n" "$*"; }
err()  { printf "${RED}[ERR]${RST} %s\n" "$*" >&2; }

# Resolve model path and served name from preset key
resolve_preset() {
    local preset="$1"
    local i
    for i in "${!PRESET_KEYS[@]}"; do
        if [[ "${PRESET_KEYS[$i]}" == "$preset" ]]; then
            RESOLVED_MODEL="${PRESET_MODELS[$i]}"
            # Served name: lowercase, replace non-alnum with dash
            RESOLVED_SERVED=$(basename "$RESOLVED_MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
            return 0
        fi
    done
    err "Unknown preset: $preset"
    return 1
}

# Launch an inference container; does not wait for health
launch_container() {
    local engine="$1" model="$2" served="$3" tp="$4" gpu_vis="$5"
    local container="${engine}_server"

    docker rm -f "$container" 2>/dev/null || true

    local rocr_env=()
    [[ -n "$gpu_vis" ]] && rocr_env=(-e "ROCR_VISIBLE_DEVICES=${gpu_vis}")

    log "Launching $engine container (tp=$tp${gpu_vis:+, GPUs=$gpu_vis})..."

    if [[ "$engine" == "sglang" ]]; then
        docker run -d \
            --name "$container" \
            --device=/dev/kfd --device=/dev/dri \
            --ipc=host --shm-size 32G \
            --group-add video --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            -p "${PORT}:${PORT}" \
            -v "${HF_CACHE}:/root/.cache/huggingface" \
            "${rocr_env[@]}" \
            "$SGLANG_IMAGE" \
            python3 -m sglang.launch_server \
                --model-path "$model" \
                --served-model-name "$served" \
                --host 0.0.0.0 \
                --port "$PORT" \
                --tp-size "$tp" \
                --api-key "$API_KEY" \
                --mem-fraction-static 0.85 \
                --attention-backend triton \
                --reasoning-parser qwen3 \
                --tool-call-parser qwen3_coder \
                --trust-remote-code

    elif [[ "$engine" == "vllm" ]]; then
        docker run -d \
            --name "$container" \
            --device=/dev/kfd --device=/dev/dri \
            --ipc=host --shm-size 32G \
            --group-add video --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            -p "${PORT}:${PORT}" \
            -v "${HF_CACHE}:/root/.cache/huggingface" \
            -e VLLM_ROCM_USE_AITER=1 \
            "${rocr_env[@]}" \
            "$VLLM_IMAGE" \
                --model "$model" \
                --served-model-name "$served" \
                --host 0.0.0.0 \
                --port "$PORT" \
                --tensor-parallel-size "$tp" \
                --api-key "$API_KEY" \
                --reasoning-parser qwen3 \
                --enable-auto-tool-choice \
                --tool-call-parser hermes \
                --trust-remote-code \
                --enable-prefix-caching
    fi
}

# Wait for the /health endpoint to respond, checking container state
wait_for_health() {
    local container="$1"
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    local last_log=$(date +%s)
    local elapsed=0

    while [[ $(date +%s) -lt $deadline ]]; do
        if curl -sf "http://localhost:${PORT}/health" &>/dev/null; then
            elapsed=$(( $(date +%s) - (deadline - HEALTH_TIMEOUT) ))
            ok "Server ready (${elapsed}s elapsed)"
            return 0
        fi
        local state
        state=$(docker inspect --format '{{.State.Status}}' "$container" 2>/dev/null || echo "gone")
        if [[ "$state" != "running" ]]; then
            err "Container stopped unexpectedly (state=$state)"
            return 1
        fi
        local now=$(date +%s)
        if (( now - last_log >= 60 )); then
            elapsed=$(( now - (deadline - HEALTH_TIMEOUT) ))
            log "  Still loading... ${elapsed}s elapsed"
            last_log=$now
        fi
        sleep 10
    done
    err "Health check timed out after ${HEALTH_TIMEOUT}s"
    return 1
}

# Tear down container and wait for VRAM to clear
teardown() {
    local engine="$1"
    local container="${engine}_server"
    log "Stopping container $container..."
    docker rm -f "$container" 2>/dev/null || true
    log "Waiting ${COOLDOWN}s for VRAM to clear..."
    sleep "$COOLDOWN"
}

# Write an error/OOM result JSON
write_error_result() {
    local result_file="$1" preset="$2" engine="$3" status="$4" error_msg="$5"
    python3 - <<PYEOF
import json
with open('$result_file', 'w') as f:
    json.dump({
        "preset": "$preset",
        "engine": "$engine",
        "status": "$status",
        "error": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$error_msg"),
    }, f, indent=2)
PYEOF
}

# =============================================================================
# Main loop
# =============================================================================
ENGINES=("sglang" "vllm")

log "============================================================"
log " OpenClaw AMD — Benchmark runner"
log " HF cache : $HF_CACHE"
log " Results  : $RESULTS_DIR"
log "============================================================"

TOTAL=0; DONE=0; SKIPPED=0; FAILED=0

for engine in "${ENGINES[@]}"; do
    [[ -n "$ONLY_ENGINE" && "$ONLY_ENGINE" != "$engine" ]] && continue

    for preset in "${PRESET_KEYS[@]}"; do
        [[ -n "$ONLY_PRESET" && "$ONLY_PRESET" != "$preset" ]] && continue

        TOTAL=$(( TOTAL + 1 ))
        result_file="$RESULTS_DIR/${engine}-${preset}.json"

        # Skip if already completed successfully
        if [[ -f "$result_file" ]]; then
            existing_status=$(python3 -c "import json; print(json.load(open('$result_file')).get('status','?'))" 2>/dev/null || echo "?")
            if [[ "$existing_status" == "ok" ]]; then
                log "SKIP (done): engine=$engine preset=$preset"
                SKIPPED=$(( SKIPPED + 1 ))
                continue
            fi
        fi

        log ""
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log " engine=$engine  preset=$preset"
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        resolve_preset "$preset" || { FAILED=$(( FAILED + 1 )); continue; }
        model="$RESOLVED_MODEL"
        served="$RESOLVED_SERVED"
        tp="${PRESET_TP[$preset]:-1}"
        gpu_vis="${PRESET_GPU_VIS[$preset]:-}"

        launch_container "$engine" "$model" "$served" "$tp" "$gpu_vis"
        container="${engine}_server"

        if ! wait_for_health "$container"; then
            error_msg=$(docker logs "$container" --tail 10 2>&1 | tr '\n"' '  ')
            warn "Container failed to become healthy"
            write_error_result "$result_file" "$preset" "$engine" "oom" "$error_msg"
            teardown "$engine"
            FAILED=$(( FAILED + 1 ))
            continue
        fi

        log "Running bench.py (concurrency: $CONCURRENCY, ${REQUESTS} req each)..."
        bench_out=$(python3 "$SCRIPT_DIR/bench.py" \
            --base-url "http://localhost:${PORT}/v1" \
            --model "$served" \
            --preset "$preset" \
            --engine "$engine" \
            --api-key "$API_KEY" \
            --concurrency $CONCURRENCY \
            --requests "$REQUESTS" \
            --max-tokens 200 \
            --output json 2>/tmp/bench_stderr.txt)
        bench_exit=$?

        if [[ $bench_exit -eq 0 ]] && echo "$bench_out" | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; then
            echo "$bench_out" > "$result_file"
            ok "Benchmark saved: $result_file"
            DONE=$(( DONE + 1 ))
        else
            stderr_tail=$(tail -3 /tmp/bench_stderr.txt | tr '\n"' '  ')
            err "bench.py failed (exit=$bench_exit)"
            write_error_result "$result_file" "$preset" "$engine" "error" "$stderr_tail"
            FAILED=$(( FAILED + 1 ))
        fi

        teardown "$engine"
    done
done

# =============================================================================
# Summary
# =============================================================================
log ""
log "============================================================"
log " Benchmark run complete"
log "  Total    : $TOTAL"
ok "  Done     : $DONE"
[[ $SKIPPED -gt 0 ]] && log "  Skipped  : $SKIPPED (already complete)"
[[ $FAILED -gt 0 ]]  && err "  Failed   : $FAILED"
log "============================================================"

if ! $SKIP_FORMAT; then
    log ""
    log "Formatting results → README.md ..."
    python3 "$SCRIPT_DIR/format_results.py" "$RESULTS_DIR"
fi
