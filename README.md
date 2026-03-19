# OpenClaw + SGLang on AMD Instinct MI300X

One command to launch a [SGLang](https://github.com/sgl-project/sglang) inference server with Qwen3.5-122B and install [OpenClaw](https://openclaw.ai) fully configured against it — no manual steps.

## Quick Start

```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/main/setup_openclaw_sglang.sh | bash
```

That's it. The script will:

1. Start the SGLang Docker container (`Qwen/Qwen3.5-122B-A10B-FP8`, port 8090)
2. Wait for the server to be ready — including Docker image pull and model load (up to 1 hour)
3. Install OpenClaw via npm (installs Node 22 if needed)
4. Configure OpenClaw to point at the SGLang endpoint automatically
5. Start the OpenClaw gateway
6. Print the SSH tunnel command to access the browser UI

## Requirements

- AMD Instinct MI300X GPU
- ROCm drivers installed (`/dev/kfd`, `/dev/dri` available)
- Docker
- Internet access (for Docker image + model weights on first run)

## Options

Pass flags after the script:

```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/main/setup_openclaw_sglang.sh | bash -s -- [options]
```

| Option | Default | Description |
|---|---|---|
| `--port PORT` | `8090` | Port to expose the SGLang server on |
| `--api-key KEY` | `abc-123` | API key for the SGLang endpoint |
| `--model MODEL_PATH` | `Qwen/Qwen3.5-122B-A10B-FP8` | HuggingFace model path |
| `--served-name NAME` | `qwen3-5-122b` | Model name used in API calls |
| `--hf-cache PATH` | `/data/hf_cache` | Local path to cache model weights |
| `--tp-size N` | `1` | Tensor parallel size |
| `--no-wait` | — | Skip waiting for SGLang to be ready before installing OpenClaw |
| `--sglang-only` | — | Only start SGLang, skip OpenClaw install |
| `--openclaw-only` | — | Only install OpenClaw (assumes SGLang is already running) |

### Examples

Custom API key and port:
```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/main/setup_openclaw_sglang.sh | bash -s -- --api-key my-secret-key --port 8080
```

SGLang only (skip OpenClaw):
```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/main/setup_openclaw_sglang.sh | bash -s -- --sglang-only
```

OpenClaw only (SGLang already running):
```bash
curl -fsSL https://raw.githubusercontent.com/Mahdi-CV/openclaw-amd-sglang/main/setup_openclaw_sglang.sh | bash -s -- --openclaw-only
```

## After Setup

The script prints a ready-to-use dashboard URL with the auth token included. To open the browser UI:

1. On your **local machine**, open the SSH tunnel:
   ```bash
   ssh -N -L 18789:127.0.0.1:18789 root@<server-ip>
   ```
2. Open the URL printed by the script (it includes the token), or run on the server:
   ```bash
   openclaw dashboard
   ```

Other useful commands:
```bash
# Check OpenClaw status
openclaw status

# Tail SGLang logs
docker logs -f sglang_server

# Tail gateway logs
tail -f /tmp/openclaw-gateway.log

# Stop everything
docker rm -f sglang_server
openclaw gateway stop
```

## Related

- [OpenClaw on AMD Developer Cloud: Qwen 3.5 and SGLang](https://www.amd.com/en/developer/resources/technical-articles/2026/openclaw-on-amd-developer-cloud-qwen-3-5-and-sglang.html)
- [OpenClaw Docs](https://docs.openclaw.ai)
- [SGLang](https://github.com/sgl-project/sglang)
