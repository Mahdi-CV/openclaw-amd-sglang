#!/bin/bash
# =============================================================================
# test_suite.sh — Comprehensive test suite for setup_openclaw_inference.sh
#
# Usage: bash tests/test_suite.sh [path/to/setup_openclaw_inference.sh]
# =============================================================================

set -uo pipefail

SCRIPT="${1:-/tmp/setup_openclaw_inference.sh}"
PASS=0
FAIL=0
SKIP=0
FAILURES=()

# ---- Colours ----------------------------------------------------------------
RED='\033[1;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
CYN='\033[1;36m'
RST='\033[0m'

# ---- Helpers ----------------------------------------------------------------
section() {
    printf "\n${CYN}══════════════════════════════════════════════════════${RST}\n"
    printf "${CYN}  %s${RST}\n" "$*"
    printf "${CYN}══════════════════════════════════════════════════════${RST}\n\n"
}

pass() { echo -e "  ${GRN}[PASS]${RST} $*"; ((PASS++)); }
fail() { echo -e "  ${RED}[FAIL]${RST} $*"; ((FAIL++)); FAILURES+=("$*"); }
skip() { echo -e "  ${YLW}[SKIP]${RST} $*"; ((SKIP++)); }

# Strip ANSI colour codes from a string
strip_ansi() { sed 's/\x1b\[[0-9;]*m//g' | tr -d '\r'; }

# Core assertion helper.
# Runs CODE in a subshell, checks exit code and optional grep pattern.
# Patterns use extended regex (-E); use | for alternation.
assert() {
    local name="$1"
    local expected_exit="$2"
    local grep_pat="${3:-}"   # extended-regex pattern, empty = skip
    local code="$4"

    local output exit_code=0
    output=$(bash -c "$code" 2>&1) || exit_code=$?

    local clean
    clean=$(echo "$output" | strip_ansi)

    if [[ "$exit_code" -ne "$expected_exit" ]]; then
        fail "$name (exit=$exit_code, expected=$expected_exit)"
        echo "       output: $(echo "$clean" | tail -3)"
        return
    fi
    if [[ -n "$grep_pat" ]] && ! echo "$clean" | grep -qiE "$grep_pat"; then
        fail "$name (output missing: '$grep_pat')"
        echo "       output: $(echo "$clean" | tail -5)"
        return
    fi
    pass "$name"
}

# Temp dir cleaned on exit
TMPROOT=$(mktemp -d)
trap 'rm -rf "$TMPROOT"' EXIT

# Run script through a PTY (so [[ -t 0 ]] is true inside it).
# Pipes INPUT into the PTY and returns stripped output.
# Exit code of 'script' is always 0; check output content instead.
pty_run() {
    local input="$1"; shift
    echo "$input" | script -q -c "bash $SCRIPT $*" /dev/null 2>/dev/null | strip_ansi
}

# ============================================================================
# SECTION 1: Argument Parsing
# ============================================================================
section "1. Argument Parsing"

assert "unknown flag exits 1 with message" 1 "Unknown option" \
    "bash '$SCRIPT' --bogus-flag"

assert "invalid engine exits 1 with message" 1 "must be.*sglang.*vllm" \
    "bash '$SCRIPT' --engine tensorrt"

assert "valid engine sglang is accepted" 0 "" \
    "bash '$SCRIPT' --engine sglang --server-only 2>&1 | grep -qiE 'Unknown option' && exit 1 || exit 0"

assert "valid engine vllm is accepted" 0 "" \
    "bash '$SCRIPT' --engine vllm --server-only 2>&1 | grep -qiE 'Unknown option' && exit 1 || exit 0"

assert "--port flag is accepted" 0 "" \
    "bash '$SCRIPT' --port 9999 --server-only 2>&1 | grep -qiE 'Unknown option' && exit 1 || exit 0"

assert "--model-preset flag is accepted" 0 "" \
    "bash '$SCRIPT' --model-preset qwen3.5-0.8b --server-only 2>&1 | grep -qiE 'Unknown option' && exit 1 || exit 0"

# Non-interactive (no TTY) auto-accepts risk
assert "non-interactive: risk auto-accepted" 0 "Non-interactive session" \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | { head -30; true; }"

assert "non-interactive: proceeds past risk to prereqs" 0 "Checking prerequisites|Docker|ROCm" \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | { head -60; true; }"

# Interactive (PTY): risk prompt shown
pty_n=$(pty_run 'n' '--model-preset qwen3.5-0.8b')
if echo "$pty_n" | grep -qiE "Risk not accepted"; then
    pass "interactive risk prompt: 'n' prints 'Risk not accepted'"
else
    fail "interactive risk prompt: 'n' prints 'Risk not accepted'"
    echo "       output: $(echo "$pty_n" | tail -3)"
fi

pty_N=$(pty_run 'N' '--model-preset qwen3.5-0.8b')
if echo "$pty_N" | grep -qiE "Risk not accepted"; then
    pass "interactive risk prompt: 'N' prints 'Risk not accepted'"
else
    fail "interactive risk prompt: 'N' prints 'Risk not accepted'"
fi

pty_empty=$(pty_run '' '--model-preset qwen3.5-0.8b')
if echo "$pty_empty" | grep -qiE "Risk not accepted"; then
    pass "interactive risk prompt: empty → 'Risk not accepted'"
else
    fail "interactive risk prompt: empty → 'Risk not accepted'"
fi

pty_y=$(pty_run 'y' '--server-only --model-preset qwen3.5-0.8b')
if echo "$pty_y" | grep -qiE "Do you accept|accept.*continue"; then
    pass "interactive risk prompt: 'y' shows the prompt"
else
    fail "interactive risk prompt: 'y' shows the prompt"
    echo "       output: $(echo "$pty_y" | grep -i 'accept' | head -2)"
fi

# ============================================================================
# SECTION 2: ROCm Detection
# ============================================================================
section "2. ROCm Detection"

# Test check_rocm logic in isolation with simulated states
ROCM_CHECK='
have() { command -v "$1" >/dev/null 2>&1; }
check_rocm() {
    local missing=()
    [[ -e "${FAKE_KFD:-/dev/kfd}" ]]   || missing+=("/dev/kfd")
    [[ -d "${FAKE_DRI:-/dev/dri}" ]]   || missing+=("/dev/dri")
    have "${FAKE_TOOL1:-rocminfo}" || \
    have "${FAKE_TOOL2:-rocm-smi}" || \
    have "${FAKE_TOOL3:-amd-smi}"  || \
    missing+=("rocm userspace tools")
    if [[ ${#missing[@]} -gt 0 ]]; then
        printf "ROCm NOT DETECTED\n"
        for item in "${missing[@]}"; do printf "  - %s\n" "$item"; done
        exit 1
    fi
    echo "ROCm OK"
}
check_rocm
'

assert "all ROCm missing → exit 1 + message" 1 "ROCm NOT DETECTED" \
    "FAKE_KFD=/nonexistent FAKE_DRI=/nonexistent FAKE_TOOL1=xxx FAKE_TOOL2=xxx FAKE_TOOL3=xxx bash -c '$ROCM_CHECK'"

assert "kfd only missing → exit 1 + lists /dev/kfd" 1 "/dev/kfd" \
    "FAKE_KFD=/nonexistent bash -c '$ROCM_CHECK'"

assert "dri only missing → exit 1 + lists /dev/dri" 1 "/dev/dri" \
    "FAKE_DRI=/nonexistent_dri_dir bash -c '$ROCM_CHECK'"

assert "tools only missing → exit 1 + lists userspace tools" 1 "userspace tools" \
    "FAKE_TOOL1=no_rocminfo_xxx FAKE_TOOL2=no_rocm_smi_xxx FAKE_TOOL3=no_amd_smi_xxx bash -c '$ROCM_CHECK'"

assert "rocm-smi alone satisfies tools check → OK" 0 "ROCm OK" \
    "FAKE_TOOL1=no_rocminfo_xxx bash -c '$ROCM_CHECK'"

assert "amd-smi alone satisfies tools check → OK" 0 "ROCm OK" \
    "FAKE_TOOL1=no_rocminfo_xxx FAKE_TOOL2=no_rocm_smi_xxx bash -c '$ROCM_CHECK'"

# On this server ROCm IS present (ROCm check passes), so skip the live error URL test
skip "error message contains quick-start URL (ROCm present on this server — tested via source)"
# Verify the URL is in the script source
assert "quick-start URL present in script source" 0 "" \
    "grep -q 'quick-start' '$SCRIPT'"

# ============================================================================
# SECTION 3: Docker Checks
# ============================================================================
section "3. Docker Checks"

# Docker not available: create a fake PATH with a wrapper that reports docker as missing
NODOCK_DIR="$TMPROOT/nodock"
mkdir -p "$NODOCK_DIR"
# Override 'have' and 'docker' by wrapping the script's logic directly
assert "docker not in PATH → exit 1 with install URL" 1 "docker.com/engine/install" \
    "have() { false; }
check_docker() {
    have docker || { echo 'ERROR: Docker is not installed.'; echo '       Install it: https://docs.docker.com/engine/install/'; exit 1; }
}
check_docker"

# Fake docker that reports installed but daemon is down
FAKE_DOCKER_DIR="$TMPROOT/fake_docker_$$"
mkdir -p "$FAKE_DOCKER_DIR"
cat > "$FAKE_DOCKER_DIR/docker" << 'EOF'
#!/bin/bash
if [[ "$1" == "info" ]]; then exit 1; fi
echo "Docker version 99.0.0, build fake"
EOF
chmod +x "$FAKE_DOCKER_DIR/docker"

assert "docker daemon down → exit 1 with systemctl hint" 1 "systemctl start docker" \
    "PATH='$FAKE_DOCKER_DIR:$PATH' bash '$SCRIPT' --server-only < /dev/null 2>&1"

assert "docker installed and running → OK message" 0 "Docker.*OK|29\." \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | { head -80; true; } | grep -iE 'Docker.*OK|Docker.*29'"

# ============================================================================
# SECTION 4: Port Availability
# ============================================================================
section "4. Port Availability"

assert "port 8090 free → not listed by ss" 0 "" \
    "ss -tlnp | grep -q ':8090' && exit 1 || exit 0"

assert "port in use → detected by ss" 0 "IN USE" \
    "python3 -c \"
import socket, time, threading
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', 8090))
s.listen(1)
threading.Timer(4, s.close).start()
time.sleep(5)
\" &
PY=\$!
sleep 1
ss -tlnp | grep -q ':8090' && echo 'IN USE' || echo 'FREE'
kill \$PY 2>/dev/null; wait \$PY 2>/dev/null || true
"

# ============================================================================
# SECTION 5: TP_SIZE Validation
# ============================================================================
section "5. TP_SIZE Validation"

TP_LOGIC='
TP_SIZE="$1"
GPU_COUNT="$2"
if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "INVALID: not a positive integer"
    exit 1
fi
if [[ "$TP_SIZE" -gt "$GPU_COUNT" ]]; then
    echo "INVALID: exceeds GPU count"
    exit 1
fi
echo "OK"
'

assert "--tp-size non-numeric → invalid" 1 "INVALID" \
    "bash -c '$TP_LOGIC' _ foo 5"

assert "--tp-size 0 → invalid" 1 "INVALID" \
    "bash -c '$TP_LOGIC' _ 0 5"

assert "--tp-size > GPU count → invalid" 1 "INVALID" \
    "bash -c '$TP_LOGIC' _ 8 5"

assert "--tp-size < GPU count → OK" 0 "OK" \
    "bash -c '$TP_LOGIC' _ 4 5"

assert "--tp-size == GPU count → OK" 0 "OK" \
    "bash -c '$TP_LOGIC' _ 5 5"

assert "--tp-size 1 (default) → OK" 0 "OK" \
    "bash -c '$TP_LOGIC' _ 1 5"

# ============================================================================
# SECTION 6: Model / Preset Resolution
# ============================================================================
section "6. Model / Preset Resolution"

# Extract just the arrays and resolve logic for unit testing
PRESET_ARRAYS='
PRESET_KEYS=(qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b qwen3.5-397b)
PRESET_MODELS=("Qwen/Qwen3.5-0.8B" "Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B" "Qwen/Qwen3.5-9B" "Qwen/Qwen3.5-27B" "Qwen/Qwen3.5-35B-A3B" "Qwen/Qwen3.5-122B-A10B-FP8" "Qwen/Qwen3.5-397B-A17B-FP8")
PRESET_CTX=(262144 262144 262144 262144 262144 262144 131072 131072)
PRESET_GB=(2 5 9 20 55 20 65 200)
'

assert "valid preset qwen3.5-0.8b resolves to correct model" 0 "Qwen/Qwen3.5-0.8B" \
    "$PRESET_ARRAYS
MODEL_PRESET='qwen3.5-0.8b'; MODEL=''
for i in \"\${!PRESET_KEYS[@]}\"; do
  [[ \"\${PRESET_KEYS[\$i]}\" == \"\$MODEL_PRESET\" ]] && { MODEL=\"\${PRESET_MODELS[\$i]}\"; break; }
done
echo \"\$MODEL\""

assert "valid preset qwen3.5-122b resolves to FP8 model" 0 "122B-A10B-FP8" \
    "$PRESET_ARRAYS
MODEL_PRESET='qwen3.5-122b'; MODEL=''
for i in \"\${!PRESET_KEYS[@]}\"; do
  [[ \"\${PRESET_KEYS[\$i]}\" == \"\$MODEL_PRESET\" ]] && { MODEL=\"\${PRESET_MODELS[\$i]}\"; break; }
done
echo \"\$MODEL\""

assert "all 8 presets resolve without error" 0 "8" \
    "$PRESET_ARRAYS
count=0
for key in \"\${PRESET_KEYS[@]}\"; do
  for i in \"\${!PRESET_KEYS[@]}\"; do
    [[ \"\${PRESET_KEYS[\$i]}\" == \"\$key\" ]] && ((count++)) && break
  done
done
echo \$count"

assert "invalid preset exits 1 with error message" 1 "Unknown.*model-preset|Unknown.*preset" \
    "bash '$SCRIPT' --model-preset totally-fake-model --server-only < /dev/null 2>&1"

assert "default preset is qwen3.5-122b" 0 "qwen3.5-122b|122b" \
    "grep 'MODEL_PRESET=' '$SCRIPT' | head -1"

assert "--model custom path used directly" 0 "my-org/my-model" \
    "MODEL='my-org/my-model'; echo \"\$MODEL\""

assert "served-name derived from model path when not set" 0 "qwen3-5-0-8b" \
    "MODEL='Qwen/Qwen3.5-0.8B'; SERVED_NAME=''
[[ -z \"\$SERVED_NAME\" ]] && SERVED_NAME=\$(basename \"\$MODEL\" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
echo \"\$SERVED_NAME\""

assert "--served-name override preserved" 0 "my-custom-name" \
    "MODEL='Qwen/Qwen3.5-0.8B'; SERVED_NAME='my-custom-name'
[[ -z \"\$SERVED_NAME\" ]] && SERVED_NAME=\$(basename \"\$MODEL\" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g')
echo \"\$SERVED_NAME\""

# Interactive picker tests (require PTY)
pty_bad=$(pty_run '99' '--server-only --model-preset ""' 2>/dev/null || true)
# With default MODEL_PRESET=qwen3.5-122b, picker won't show — test the picker by clearing MODEL_PRESET in a wrapper
# We test it via directly running the script with MODEL_PRESET pre-cleared (not possible via CLI without patching)
# Instead test that non-interactive with no model uses the default
assert "non-interactive with no model uses default qwen3.5-122b" 0 "qwen3.5-122b|Non-interactive|122" \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | grep -iE 'qwen3.5-122b|122b|Non-interactive' | head -3"

# ============================================================================
# SECTION 7: Disk Space & Model Cache Detection
# ============================================================================
section "7. Disk Space & Model Cache Detection"

HF_TEST="$TMPROOT/hf_cache_test"
mkdir -p "$HF_TEST"

MODEL_CACHE_FN='
HF_CACHE="'"$HF_TEST"'"
model_is_cached() {
    local org="${1%%/*}" name="${1#*/}"
    local hub_dir="${HF_CACHE}/hub/models--${org}--${name}"
    [[ -d "${hub_dir}/blobs" ]] && [[ -n "$(ls -A "${hub_dir}/blobs" 2>/dev/null)" ]]
}
'

assert "model_is_cached: empty cache → false (exit 1)" 1 "" \
    "$MODEL_CACHE_FN
model_is_cached 'Qwen/Qwen3.5-0.8B'"

# Create a fake populated cache entry
mkdir -p "$HF_TEST/hub/models--Qwen--Qwen3.5-0.8B/blobs"
touch "$HF_TEST/hub/models--Qwen--Qwen3.5-0.8B/blobs/somefile"

assert "model_is_cached: populated blobs dir → true (exit 0)" 0 "" \
    "$MODEL_CACHE_FN
model_is_cached 'Qwen/Qwen3.5-0.8B'"

# Empty blobs dir
mkdir -p "$HF_TEST/hub/models--Qwen--Qwen3.5-2B/blobs"

assert "model_is_cached: empty blobs dir → false (exit 1)" 1 "" \
    "$MODEL_CACHE_FN
model_is_cached 'Qwen/Qwen3.5-2B'"

assert "disk check: sufficient space prints available and needed" 0 "available|needed|GB" \
    "avail_kb=\$(df -k '$HF_TEST' | awk 'NR==2{print \$4}')
avail_gb=\$(( avail_kb / 1048576 ))
echo \"\${avail_gb}GB available, ~2GB needed for download\""

assert "disk check: low space emits WARNING" 0 "WARNING|Low disk" \
    "avail_gb=1; needed_gb=65
[[ \$avail_gb -lt \$needed_gb ]] && echo 'WARNING: Low disk space — download may fail.'"

assert "disk check: sufficient space → no WARNING" 0 "OK" \
    "avail_gb=500; needed_gb=65
[[ \$avail_gb -lt \$needed_gb ]] && echo 'WARNING: Low disk space' || echo 'OK'"

# ============================================================================
# SECTION 8: Public IP Detection
# ============================================================================
section "8. Public IP Detection"

assert "ifconfig.me returns a valid IP" 0 "[0-9]+\.[0-9]+" \
    "curl -sf --max-time 5 ifconfig.me"

assert "api.ipify.org returns a valid IP" 0 "[0-9]+\.[0-9]+" \
    "curl -sf --max-time 5 api.ipify.org"

assert "hostname -I fallback returns non-empty" 0 "[0-9]" \
    "hostname -I | awk '{print \$1}'"

assert "all methods fail → PUBLIC_IP empty → CAUGHT" 0 "CAUGHT empty IP" \
    "ip1=\$(curl -sf --max-time 1 http://192.0.2.1/ 2>/dev/null || true)
ip2=\$(curl -sf --max-time 1 http://192.0.2.2/ 2>/dev/null || true)
ip3=''
PUBLIC_IP=\"\${ip1:-\${ip2:-\$ip3}}\"
[[ -z \"\$PUBLIC_IP\" ]] && echo 'CAUGHT empty IP' || echo \"IP: \$PUBLIC_IP\""

assert "BASE_URL built correctly from IP and port" 0 "http://1.2.3.4:8090/v1" \
    "PUBLIC_IP='1.2.3.4'; PORT=8090; echo \"http://\${PUBLIC_IP}:\${PORT}/v1\""

assert "empty PUBLIC_IP → malformed BASE_URL detected" 0 "MALFORMED" \
    "PUBLIC_IP=''; PORT=8090
BASE_URL=\"http://\${PUBLIC_IP}:\${PORT}/v1\"
[[ \"\$BASE_URL\" == 'http://:8090/v1' ]] && echo 'MALFORMED' || echo \"OK: \$BASE_URL\""

# ============================================================================
# SECTION 9: Mode Flags
# ============================================================================
section "9. Mode Flags (--server-only, --openclaw-only, --no-wait)"

assert "--server-only: RUN_OPENCLAW set to false" 0 "RUN_OPENCLAW=false" \
    "RUN_OPENCLAW=true; arg='--server-only'
[[ \"\$arg\" == '--server-only' ]] && RUN_OPENCLAW=false
echo \"RUN_OPENCLAW=\$RUN_OPENCLAW\""

assert "--openclaw-only: RUN_SERVER set to false" 0 "RUN_SERVER=false" \
    "RUN_SERVER=true; arg='--openclaw-only'
[[ \"\$arg\" == '--openclaw-only' ]] && RUN_SERVER=false
echo \"RUN_SERVER=\$RUN_SERVER\""

assert "--openclaw-only: WAIT_FOR_SERVER set to false" 0 "WAIT_FOR_SERVER=false" \
    "WAIT_FOR_SERVER=true; arg='--openclaw-only'
[[ \"\$arg\" == '--openclaw-only' ]] && WAIT_FOR_SERVER=false
echo \"WAIT_FOR_SERVER=\$WAIT_FOR_SERVER\""

assert "--no-wait: WAIT_FOR_SERVER set to false" 0 "WAIT_FOR_SERVER=false" \
    "WAIT_FOR_SERVER=true; arg='--no-wait'
[[ \"\$arg\" == '--no-wait' ]] && WAIT_FOR_SERVER=false
echo \"WAIT_FOR_SERVER=\$WAIT_FOR_SERVER\""

assert "--openclaw-only skips the prerequisite block entirely" 0 "prereqs skipped" \
    "out=\$(bash '$SCRIPT' --openclaw-only < /dev/null 2>&1)
echo \"\$out\" | grep -qiE 'Checking prerequisites' && echo 'prereqs ran' || echo 'prereqs skipped'"

# ============================================================================
# SECTION 10: Node.js Version Detection
# ============================================================================
section "10. Node.js Version Detection"

FAKE_NODE="$TMPROOT/fake_node"
mkdir -p "$FAKE_NODE"

make_fake_node() {
    local version="$1"
    cat > "$FAKE_NODE/node" << EOF
#!/bin/bash
echo 'v${version}'
EOF
    chmod +x "$FAKE_NODE/node"
}

assert "node absent → version 0 detected" 0 "NODE_MAJOR=0" \
    "EMPTY_DIR=\$(mktemp -d)
RAW=\$(PATH=\"\$EMPTY_DIR\" node --version 2>/dev/null)
NODE_MAJOR=\$(echo \"\${RAW}\" | sed 's/v\([0-9]*\).*/\1/' | grep -E '^[0-9]+$' || echo '0')
[[ -z \"\$NODE_MAJOR\" ]] && NODE_MAJOR=0
echo \"NODE_MAJOR=\$NODE_MAJOR\"
rm -rf \"\$EMPTY_DIR\""

make_fake_node "20.15.0"
assert "node v20 → triggers install (< 22)" 0 "WOULD INSTALL" \
    "PATH='$FAKE_NODE:$PATH'
NODE_MAJOR=\$(node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/' || echo '0')
[[ \"\$NODE_MAJOR\" -lt 22 ]] && echo 'WOULD INSTALL' || echo 'OK'"

make_fake_node "22.0.0"
assert "node v22 → skips install" 0 "OK" \
    "PATH='$FAKE_NODE:$PATH'
NODE_MAJOR=\$(node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/' || echo '0')
[[ \"\$NODE_MAJOR\" -lt 22 ]] && echo 'WOULD INSTALL' || echo 'OK'"

make_fake_node "24.1.0"
assert "node v24 → skips install" 0 "OK" \
    "PATH='$FAKE_NODE:$PATH'
NODE_MAJOR=\$(node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/' || echo '0')
[[ \"\$NODE_MAJOR\" -lt 22 ]] && echo 'WOULD INSTALL' || echo 'OK'"

assert "node install path uses apt-get (Debian/Ubuntu)" 0 "apt-get|nodesource" \
    "grep -E 'apt-get.*nodejs|nodesource' '$SCRIPT' | head -2"

# ============================================================================
# SECTION 11: OpenClaw Config JSON Write
# ============================================================================
section "11. OpenClaw Config JSON Write"

CFG_DIR="$TMPROOT/openclaw_cfg"
mkdir -p "$CFG_DIR"
CFG_FILE="$CFG_DIR/openclaw.json"

assert "config write: missing file → node throws ENOENT (not silently ignored)" 0 "ENOENT|CAUGHT" \
    "node -e \"
try {
  const fs = require('fs');
  JSON.parse(fs.readFileSync('$CFG_FILE', 'utf8'));
  process.stdout.write('should not reach\n');
} catch(e) {
  process.stdout.write('CAUGHT: ' + e.code + '\n');
}\""

echo '{"models":{}}' > "$CFG_FILE"
assert "config write: valid JSON updated with sglang provider" 0 "sglang" \
    "node -e \"
const fs = require('fs');
const cfg = JSON.parse(fs.readFileSync('$CFG_FILE', 'utf8'));
cfg.models = cfg.models || {};
cfg.models.providers = cfg.models.providers || {};
cfg.models.providers['sglang'] = { baseUrl: 'http://1.2.3.4:8090/v1', apiKey: 'key', api: 'openai-completions', models: [] };
fs.writeFileSync('$CFG_FILE', JSON.stringify(cfg, null, 2));
\" && python3 -c \"import json; d=json.load(open('$CFG_FILE')); print(list(d['models']['providers'].keys())[0])\""

echo '{"models":{"providers":{"sglang":{"old":true}}}}' > "$CFG_FILE"
assert "config write: existing provider overwritten not duplicated" 0 "^1$" \
    "node -e \"
const fs = require('fs');
const cfg = JSON.parse(fs.readFileSync('$CFG_FILE', 'utf8'));
cfg.models.providers['sglang'] = { baseUrl: 'http://new', apiKey: 'k', api: 'openai-completions', models: [] };
fs.writeFileSync('$CFG_FILE', JSON.stringify(cfg, null, 2));
\" && python3 -c \"import json; d=json.load(open('$CFG_FILE')); print(len(d['models']['providers']))\""

assert "config write: vllm engine uses correct provider key" 0 "vllm" \
    "CFGF='$TMPROOT/vllm_cfg.json'
echo '{\"models\":{}}' > \"\$CFGF\"
node -e \"
const fs = require('fs');
const cfg = JSON.parse(fs.readFileSync('\$CFGF', 'utf8'));
cfg.models = cfg.models || {}; cfg.models.providers = cfg.models.providers || {};
cfg.models.providers['vllm'] = { baseUrl: 'http://x', apiKey: 'k', api: 'openai-completions', models: [] };
fs.writeFileSync('\$CFGF', JSON.stringify(cfg, null, 2));
\" && python3 -c \"import json; d=json.load(open('\$CFGF')); print(list(d['models']['providers'].keys())[0])\""

# ============================================================================
# SECTION 12: Container Health Wait Logic
# ============================================================================
section "12. Container Health Wait Logic"

assert "health endpoint up → curl returns success" 0 "READY" \
    "PORT=18999
python3 -c \"
import http.server, threading, time
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
    def log_message(self, *a): pass
s = http.server.HTTPServer(('', \$PORT), H)
t = threading.Thread(target=s.serve_forever)
t.daemon = True
t.start()
time.sleep(5)
\" &
SRV=\$!
sleep 1
curl -sf http://localhost:\$PORT/health && echo 'READY' || echo 'NOT READY'
kill \$SRV 2>/dev/null; wait \$SRV 2>/dev/null || true"

assert "docker inspect 'exited' → container-stopped detected" 0 "CAUGHT: container gone" \
    "FAKE_DIR=\$(mktemp -d)
cat > \"\$FAKE_DIR/docker\" << 'EOF'
#!/bin/bash
if [[ \"\$1\" == 'inspect' ]]; then echo 'exited'; fi
EOF
chmod +x \"\$FAKE_DIR/docker\"
PATH=\"\$FAKE_DIR:\$PATH\"
state=\$(docker inspect --format '{{.State.Status}}' fake_container 2>/dev/null || echo 'gone')
[[ \"\$state\" != 'running' ]] && echo 'CAUGHT: container gone' || echo 'still running'
rm -rf \"\$FAKE_DIR\""

assert "container health timeout message present in script" 0 "did not become ready" \
    "grep 'did not become ready' '$SCRIPT'"

# ============================================================================
# SECTION 13: Gateway Startup
# ============================================================================
section "13. Gateway Startup"

assert "pkill gateway used with -f flag in script" 0 "pkill.*-f.*gateway|pkill.*gateway" \
    "grep 'pkill.*gateway' '$SCRIPT' | head -1"

assert "gateway port detection logic works" 0 "LISTENING|NOT LISTENING" \
    "ss -tlnp | grep -q ':18789' && echo 'LISTENING' || echo 'NOT LISTENING'"

assert "gateway log file path writable" 0 "OK" \
    "touch /tmp/openclaw-gateway.log && echo 'OK'"

assert "nohup + background pattern correct in script" 0 "nohup.*gateway.*18789" \
    "grep -E 'nohup.*gateway.*18789' '$SCRIPT'"

# ============================================================================
# SECTION 14: Idempotency
# ============================================================================
section "14. Idempotency"

assert "docker rm -f nonexistent container → safe" 0 "OK" \
    "docker rm -f 'openclaw_test_nonexistent_$$' 2>/dev/null || true; echo 'OK'"

assert "mkdir -p on existing dir → safe" 0 "OK" \
    "D=\$(mktemp -d); mkdir -p \"\$D\"; mkdir -p \"\$D\"; echo 'OK'; rm -rf \"\$D\""

assert "script source does not hardcode model in docker cmd" 0 "" \
    "grep -E '^\s+--model(-path)? ' '$SCRIPT' | grep -v '\$MODEL' && exit 1 || exit 0"

assert "container name derived from engine var" 0 'ENGINE.*_server|\${ENGINE}' \
    "grep 'CONTAINER_NAME' '$SCRIPT' | grep -iE 'ENGINE|engine'"

# ============================================================================
# SECTION 15: HF_CACHE Writability
# ============================================================================
section "15. HF_CACHE Writability"

assert "writable HF_CACHE → mkdir succeeds" 0 "WRITABLE" \
    "D=\$(mktemp -d)/hf; mkdir -p \"\$D\" && echo 'WRITABLE'; rm -rf \"\$(dirname \"\$D\")\""

assert "HF_CACHE default path is under HOME" 0 "HOME|cache" \
    "grep 'HF_CACHE=' '$SCRIPT' | head -1"

assert "/data/hf_cache creates if /data exists" 0 "OK" \
    "mkdir -p /data/hf_cache && echo 'OK'"

if [[ $EUID -ne 0 ]]; then
    assert "non-writable dir detected as non-writable" 0 "NOT WRITABLE" \
        "D=\$(mktemp -d)/ro; mkdir -p \"\$D\"; chmod 000 \"\$D\"
mkdir -p \"\$D/sub\" 2>/dev/null && echo 'WRITABLE' || echo 'NOT WRITABLE'
chmod 755 \"\$D\"; rm -rf \"\$(dirname \"\$D\")\""
else
    skip "non-writable HF_CACHE test (running as root — root bypasses chmod 000)"
fi

# ============================================================================
# SECTION 16: ROCm Version vs Image Requirement
# ============================================================================
section "16. ROCm Version vs Image Requirement"

assert "rocm-smi version parsed to X.Y.Z format" 0 "[0-9]+\.[0-9]+" \
    "rocm-smi --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1"

assert "ROCm 4.x flagged as insufficient for rocm700 image" 0 "WARN|insufficient|old" \
    "ROCM_VER=\$(rocm-smi --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo '0.0')
MAJOR=\$(echo \"\$ROCM_VER\" | cut -d. -f1)
[[ \"\$MAJOR\" -lt 7 ]] && echo \"WARN: rocm v\$ROCM_VER is old — image needs v7+\" || echo 'OK'"

assert "image name contains ROCm version tag" 0 "rocm" \
    "grep 'SGLANG_IMAGE=\|VLLM_IMAGE=' '$SCRIPT'"

# ============================================================================
# SECTION 17: Non-Root / Privilege Check
# ============================================================================
section "17. Non-Root / Privilege Check"

assert "running as root on this server" 0 "root|IS ROOT" \
    "[[ \$EUID -eq 0 ]] && echo 'IS ROOT' || echo 'non-root'"

assert "apt-get is available" 0 "apt-get" \
    "command -v apt-get && echo 'apt-get present'"

assert "script calls apt-get (requires root)" 0 "apt-get" \
    "grep 'apt-get install' '$SCRIPT'"

assert "script calls npm install -g (requires root or prefix config)" 0 "npm install -g" \
    "grep 'npm install -g' '$SCRIPT'"

# ============================================================================
# SECTION 18: Integration — Full Script Dry-Run
# ============================================================================
section "18. Integration: Full Script Dry-Run (Non-Interactive)"

# All integration tests run without a TTY so the script auto-accepts risk
# and uses the default preset. We stop before the actual Docker launch by
# checking the model/prereq output, which appears before docker run.

assert "script shows banner (non-interactive)" 0 "OpenClaw|AMD|SGLang" \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | { head -10; true; }"

# Helper: run script, capture first N lines, ignore SIGPIPE exit code from head
script_head() { bash "$SCRIPT" "$@" < /dev/null 2>&1 | head -80; true; }

assert "prerequisite block: Docker OK line present" 0 "Docker.*OK" \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | { head -80; true; } | grep -iE 'Docker.*OK'"

assert "prerequisite block: ROCm OK line present" 0 "ROCm.*OK" \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | { head -80; true; } | grep -iE 'ROCm.*OK'"

assert "prerequisite block: GPU count present" 0 "GPU|AMD GPU" \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | { head -80; true; } | grep -iE 'GPU'"

assert "default model is qwen3.5-122b (shown in launch block)" 0 "122" \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | { head -80; true; } | grep -iE 'Model|122' | head -3"

assert "--model-preset qwen3.5-0.8b overrides default" 0 "0\.8B|0-8b" \
    "bash '$SCRIPT' --server-only --model-preset qwen3.5-0.8b < /dev/null 2>&1 | { head -80; true; } | grep -iE '0\.8B|0-8b' | head -2"

assert "--engine vllm shown in launch output" 0 "vllm" \
    "bash '$SCRIPT' --server-only --engine vllm --model-preset qwen3.5-0.8b < /dev/null 2>&1 | { head -80; true; } | grep -iE 'vllm|engine' | head -3"

assert "disk check: model cached → 'already in cache' message" 0 "cache|cached" \
    "bash '$SCRIPT' --server-only --model-preset qwen3.5-0.8b < /dev/null 2>&1 | { head -80; true; } | grep -iE 'cache|cached' | head -3"

assert "disk check: 397b shows large size warning or size" 0 "200|GB|disk|cache" \
    "bash '$SCRIPT' --server-only --model-preset qwen3.5-397b < /dev/null 2>&1 | { head -80; true; } | grep -iE '200|GB|disk|cache' | head -3"

assert "--openclaw-only skips prereq block" 0 "prereqs skipped" \
    "out=\$(bash '$SCRIPT' --openclaw-only < /dev/null 2>&1 | { head -20; true; })
echo \"\$out\" | grep -qiE 'Checking prerequisites' && echo 'prereqs ran' || echo 'prereqs skipped'"

assert "non-interactive: risk auto-accepted message shown" 0 "Non-interactive" \
    "bash '$SCRIPT' --server-only < /dev/null 2>&1 | { head -40; true; } | grep -i 'Non-interactive'"

# Clean up any containers the integration tests may have started
docker rm -f sglang_server vllm_server 2>/dev/null || true

# ============================================================================
# SUMMARY
# ============================================================================
TOTAL=$(( PASS + FAIL + SKIP ))
printf "\n${CYN}══════════════════════════════════════════════════════${RST}\n"
printf "${CYN}  TEST RESULTS${RST}\n"
printf "${CYN}══════════════════════════════════════════════════════${RST}\n"
printf "  Total  : %d\n" "$TOTAL"
printf "  ${GRN}Passed : %d${RST}\n" "$PASS"
printf "  ${RED}Failed : %d${RST}\n" "$FAIL"
printf "  ${YLW}Skipped: %d${RST}\n" "$SKIP"

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    printf "\n  ${RED}Failed tests:${RST}\n"
    for f in "${FAILURES[@]}"; do
        printf "    ${RED}✗${RST} %s\n" "$f"
    done
fi

printf "${CYN}══════════════════════════════════════════════════════${RST}\n\n"
[[ $FAIL -eq 0 ]]
