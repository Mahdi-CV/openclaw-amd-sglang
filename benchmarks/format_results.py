#!/usr/bin/env python3
"""
format_results.py — Read benchmarks/results/*.json and write a Markdown
Benchmarks section to stdout. Appended to README.md by run_all.sh.

Usage:
    python3 benchmarks/format_results.py benchmarks/results/
    python3 benchmarks/format_results.py benchmarks/results/ >> README.md
"""

import glob
import json
import sys
from pathlib import Path

PRESET_ORDER = [
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "qwen3.5-122b",
    "qwen3.5-397b",
]
ENGINE_ORDER = ["sglang", "vllm"]

PRESET_LABEL = {
    "qwen3.5-0.8b":    "0.8B",
    "qwen3.5-2b":      "2B",
    "qwen3.5-4b":      "4B",
    "qwen3.5-9b":      "9B",
    "qwen3.5-27b":     "27B",
    "qwen3.5-35b-a3b": "35B-A3B (MoE)",
    "qwen3.5-122b":    "122B-A10B (MoE, FP8)",
    "qwen3.5-397b":    "397B-A17B (MoE, FP8)",
}

SGLANG_IMAGE = "lmsysorg/sglang:v0.5.9-rocm700-mi30x"
VLLM_IMAGE   = "vllm/vllm-openai-rocm:v0.15.0"


def load_results(results_dir: str) -> dict:
    """Load all result files; key = (engine, preset)."""
    data = {}
    for path in glob.glob(str(Path(results_dir) / "*.json")):
        try:
            r = json.load(open(path))
            key = (r.get("engine", ""), r.get("preset", ""))
            data[key] = r
        except Exception as e:
            print(f"# Warning: could not load {path}: {e}", file=sys.stderr)
    return data


def fmt_ms(val):
    if val is None:
        return "—"
    return f"{val:,.0f}"


def fmt_tps(val):
    if val is None:
        return "—"
    return f"{val:,.0f}"


def cell(result, *keys):
    """Safely drill into nested dict."""
    v = result
    for k in keys:
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def render(data: dict) -> str:
    lines = []

    lines.append("## Benchmarks")
    lines.append("")
    lines.append(
        "**Hardware:** 5× AMD Instinct MI300X (~191 GB VRAM each)  "
        f"**SGLang:** `{SGLANG_IMAGE}`  "
        f"**vLLM:** `{VLLM_IMAGE}`  "
        "**Prompt:** ~100 input tokens → 200 output tokens  "
        "`tp=2` for 397B (2 GPUs)"
    )
    lines.append("")

    # Table header
    lines.append(
        "| Model | Params | Engine | TTFT (ms) | Tok/s | "
        "Throughput @4 (tok/s) | Throughput @16 (tok/s) | E2E Latency (ms) |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")

    for preset in PRESET_ORDER:
        for engine in ENGINE_ORDER:
            r = data.get((engine, preset))
            params = PRESET_LABEL.get(preset, preset)

            if r is None:
                lines.append(
                    f"| `{preset}` | {params} | {engine} "
                    "| — | — | — | — | — |"
                )
                continue

            status = r.get("status", "ok")
            if status in ("oom", "error"):
                tag = "OOM" if status == "oom" else "ERR"
                lines.append(
                    f"| `{preset}` | {params} | {engine} "
                    f"| {tag} | {tag} | {tag} | {tag} | {tag} |"
                )
                continue

            ttft    = fmt_ms(cell(r, "single_request", "ttft_ms"))
            tps     = fmt_tps(cell(r, "single_request", "tokens_per_sec"))
            e2e     = fmt_ms(cell(r, "single_request", "e2e_latency_ms"))
            thr4    = fmt_tps(cell(r, "concurrent", "4", "throughput_tokens_per_sec"))
            thr16   = fmt_tps(cell(r, "concurrent", "16", "throughput_tokens_per_sec"))

            lines.append(
                f"| `{preset}` | {params} | {engine} "
                f"| {ttft} | {tps} | {thr4} | {thr16} | {e2e} |"
            )

    lines.append("")
    lines.append(
        "> TTFT and E2E latency are the median of 3 sequential runs after 1 warm-up request. "
        "Throughput (tok/s) is total output tokens divided by wall-clock time under concurrent load. "
        "Temperature=0 for all runs."
    )
    lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results-dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]
    data = load_results(results_dir)

    if not data:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(render(data))


if __name__ == "__main__":
    main()
