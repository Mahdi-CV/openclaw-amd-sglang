# OpenClaw + SGLang / vLLM on AMD Instinct MI300X

One command to launch an inference server (SGLang or vLLM) with any Qwen3.5 model and install [OpenClaw](https://openclaw.ai) fully configured against it — no manual steps.

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/multi-engine/setup_openclaw_inference.sh | bash -s -- --model-preset qwen3.5-9b
```

That's it. The script will:

1. Start the inference server Docker container (port 8090)
2. Wait for the server to be ready — including Docker image pull and model load (up to 1 hour)
3. Install OpenClaw via npm (installs Node 22 if needed)
4. Configure OpenClaw to point at the endpoint automatically
5. Start the OpenClaw gateway
6. Print the SSH tunnel command to access the browser UI

## Requirements

- AMD Instinct MI300X GPU
- ROCm drivers installed (`/dev/kfd`, `/dev/dri` available)
- Docker
- Internet access (for Docker image + model weights on first run)

## Model Presets (Qwen3.5 family)

Use `--model-preset` to select a model. The script sets the HuggingFace path, served name, and context window automatically.

| Preset | HuggingFace Model | Context |
|---|---|---|
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | 262K |
| `qwen3.5-2b` | `Qwen/Qwen3.5-2B` | 262K |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | 262K |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | 262K |
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | 262K |
| `qwen3.5-35b-a3b` | `Qwen/Qwen3.5-35B-A3B` | 262K |
| `qwen3.5-122b` | `Qwen/Qwen3.5-122B-A10B-FP8` | 131K |
| `qwen3.5-397b` | `Qwen/Qwen3.5-397B-A17B-FP8` | 131K |

For a custom model, use `--model Qwen/MyModel --served-name my-model` instead.

## Options

| Option | Default | Description |
|---|---|---|
| `--engine sglang\|vllm` | `sglang` | Inference engine |
| `--model-preset PRESET` | — | Qwen3.5 preset (see table above) |
| `--model MODEL_PATH` | — | HuggingFace model path (custom) |
| `--served-name NAME` | from preset | Model name used in API calls |
| `--port PORT` | `8090` | Port to expose the server on |
| `--api-key KEY` | `abc-123` | API key for the endpoint |
| `--hf-cache PATH` | `/data/hf_cache` | Local path to cache model weights |
| `--tp-size N` | `1` | Tensor parallel size |
| `--no-wait` | — | Skip waiting for the server before installing OpenClaw |
| `--server-only` | — | Only start the inference server, skip OpenClaw |
| `--openclaw-only` | — | Only install OpenClaw (server must already be running) |

## Examples

SGLang with Qwen3.5-9B (default engine):
```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/multi-engine/setup_openclaw_inference.sh | bash -s -- --model-preset qwen3.5-9b
```

vLLM with Qwen3.5-122B, 4-GPU tensor parallel:
```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/multi-engine/setup_openclaw_inference.sh | bash -s -- --engine vllm --model-preset qwen3.5-122b --tp-size 4
```

Server only (skip OpenClaw):
```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/multi-engine/setup_openclaw_inference.sh | bash -s -- --engine sglang --model-preset qwen3.5-27b --server-only
```

OpenClaw only (server already running):
```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/multi-engine/setup_openclaw_inference.sh | bash -s -- --engine sglang --model-preset qwen3.5-9b --openclaw-only
```

## After Setup

The script prints a ready-to-use dashboard URL with the auth token included. To open the browser UI:

1. On your **local machine**, open the SSH tunnel:
   ```bash
   ssh -N -L 18789:127.0.0.1:18789 amd@<server-ip>
   ```
2. Open the URL printed by the script (it includes the token), or run on the server:
   ```bash
   openclaw dashboard
   ```

Other useful commands:
```bash
# Check OpenClaw status
openclaw status

# Tail inference server logs
docker logs -f sglang_server   # or vllm_server

# Tail gateway logs
tail -f /tmp/openclaw-gateway.log

# Stop everything
docker rm -f sglang_server     # or vllm_server
openclaw gateway stop
```

## Related

- [OpenClaw Docs](https://docs.openclaw.ai)
- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)
