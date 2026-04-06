#!/usr/bin/env python3
"""
bench.py — Benchmark an OpenAI-compatible inference endpoint.

Measures:
  - Time to First Token (TTFT)
  - Tokens per second (single request, generation speed)
  - End-to-end latency
  - Concurrent throughput @ configurable concurrency levels

Usage:
    python3 benchmarks/bench.py \
        --base-url http://localhost:8090/v1 \
        --model qwen3.5-9b \
        --engine sglang \
        --concurrency 1 4 16 \
        --requests 16 \
        --output json
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Fixed benchmark prompt (~100 tokens)
# ---------------------------------------------------------------------------
BENCHMARK_PROMPT = (
    "Write a detailed technical explanation of how transformer attention mechanisms "
    "work, including the mathematical formulation of scaled dot-product attention "
    "and why the scaling factor is important. Cover multi-head attention and explain "
    "why multiple attention heads are used in practice. Include a brief discussion "
    "of the computational complexity."
)


# ---------------------------------------------------------------------------
# Single-request measurement
# ---------------------------------------------------------------------------
async def measure_single(
    client: "AsyncOpenAI",
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    ttft_ms = None
    output_tokens = 0
    t_start = time.perf_counter()

    try:
        async with client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.0,
        ) as stream:
            async for chunk in stream:
                # Final chunk with usage stats
                if chunk.usage is not None:
                    output_tokens = chunk.usage.completion_tokens or output_tokens
                    continue
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000

    except Exception as e:
        raise RuntimeError(f"Request failed: {e}") from e

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    # Fall back: if usage not reported, estimate from TTFT and elapsed time
    if output_tokens == 0:
        output_tokens = max(1, int((elapsed - (ttft_ms or 0) / 1000) * 50))

    generation_time = elapsed - (ttft_ms or 0) / 1000
    tok_per_sec = output_tokens / generation_time if generation_time > 0 else 0.0

    return {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "e2e_latency_ms": round(elapsed * 1000, 1),
        "tokens_per_sec": round(tok_per_sec, 1),
        "output_tokens": output_tokens,
    }


# ---------------------------------------------------------------------------
# Concurrent throughput measurement
# ---------------------------------------------------------------------------
async def measure_concurrent(
    client: "AsyncOpenAI",
    model: str,
    prompt: str,
    max_tokens: int,
    n_concurrent: int,
    n_requests: int,
) -> dict:
    sem = asyncio.Semaphore(n_concurrent)

    async def bounded_request():
        async with sem:
            return await measure_single(client, model, prompt, max_tokens)

    t_wall_start = time.perf_counter()
    tasks = [asyncio.create_task(bounded_request()) for _ in range(n_requests)]
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    wall_time = time.perf_counter() - t_wall_start

    valid = [r for r in raw if isinstance(r, dict)]
    errors = len(raw) - len(valid)

    if not valid:
        return {"error": "all requests failed", "n_concurrent": n_concurrent}

    total_tokens = sum(r["output_tokens"] for r in valid)
    e2e_latencies = sorted(r["e2e_latency_ms"] for r in valid)
    n = len(e2e_latencies)
    p50 = statistics.median(e2e_latencies)
    p95 = e2e_latencies[max(0, int(n * 0.95) - 1)] if n >= 2 else e2e_latencies[-1]

    return {
        "n_concurrent": n_concurrent,
        "total_requests": len(valid),
        "errors": errors,
        "total_tokens": total_tokens,
        "wall_time_sec": round(wall_time, 2),
        "throughput_tokens_per_sec": round(total_tokens / wall_time, 1),
        "p50_e2e_ms": round(p50, 1),
        "p95_e2e_ms": round(p95, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible endpoint")
    p.add_argument("--base-url", required=True, help="Endpoint base URL, e.g. http://localhost:8090/v1")
    p.add_argument("--model", required=True, help="Model name as served by the endpoint")
    p.add_argument("--engine", required=True, choices=["sglang", "vllm"], help="Engine label for output")
    p.add_argument("--api-key", default="abc-123")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16],
                   help="Concurrency levels to test (default: 1 4 16)")
    p.add_argument("--requests", type=int, default=16,
                   help="Number of requests per concurrency level (default: 16)")
    p.add_argument("--max-tokens", type=int, default=200, help="Max output tokens (default: 200)")
    p.add_argument("--warmup", type=int, default=1, help="Warm-up requests (default: 1)")
    p.add_argument("--single-runs", type=int, default=3,
                   help="Repeated single-request runs for median (default: 3)")
    p.add_argument("--output", choices=["json", "table"], default="json")
    p.add_argument("--preset", default=None, help="Preset name (for output metadata)")
    return p.parse_args()


async def run(args):
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        max_retries=0,
        timeout=600.0,
    )

    log = lambda msg: print(f"  {msg}", file=sys.stderr)

    # --- Warm-up ---
    log(f"Warming up ({args.warmup} request(s))...")
    for i in range(args.warmup):
        try:
            await measure_single(client, args.model, BENCHMARK_PROMPT, args.max_tokens)
            log(f"  warm-up {i+1}/{args.warmup} done")
        except Exception as e:
            print(f"FATAL: warm-up failed: {e}", file=sys.stderr)
            sys.exit(1)

    # --- Single-request baseline ---
    log(f"Single-request baseline ({args.single_runs} runs)...")
    single_runs = []
    for i in range(args.single_runs):
        try:
            r = await measure_single(client, args.model, BENCHMARK_PROMPT, args.max_tokens)
            single_runs.append(r)
            log(f"  run {i+1}: TTFT={r['ttft_ms']}ms  tok/s={r['tokens_per_sec']}  e2e={r['e2e_latency_ms']}ms")
        except Exception as e:
            log(f"  run {i+1} failed: {e}")

    if not single_runs:
        print("FATAL: all single-request runs failed", file=sys.stderr)
        sys.exit(1)

    single_result = {
        "ttft_ms": round(statistics.median(r["ttft_ms"] for r in single_runs if r["ttft_ms"] is not None), 1),
        "e2e_latency_ms": round(statistics.median(r["e2e_latency_ms"] for r in single_runs), 1),
        "tokens_per_sec": round(statistics.median(r["tokens_per_sec"] for r in single_runs), 1),
        "output_tokens": single_runs[0]["output_tokens"],
    }

    # --- Concurrent throughput ---
    concurrent_results = {}
    for c in args.concurrency:
        n_req = max(args.requests, c)
        log(f"Concurrent test: {c} concurrent, {n_req} total requests...")
        try:
            result = await measure_concurrent(
                client, args.model, BENCHMARK_PROMPT, args.max_tokens,
                n_concurrent=c, n_requests=n_req,
            )
            concurrent_results[str(c)] = result
            log(f"  throughput={result.get('throughput_tokens_per_sec')} tok/s  "
                f"p50={result.get('p50_e2e_ms')}ms  p95={result.get('p95_e2e_ms')}ms")
        except Exception as e:
            log(f"  concurrent@{c} failed: {e}")
            concurrent_results[str(c)] = {"error": str(e), "n_concurrent": c}

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset or args.model,
        "model": args.model,
        "engine": args.engine,
        "base_url": args.base_url,
        "status": "ok",
        "benchmark_config": {
            "prompt_tokens_approx": 100,
            "max_output_tokens": args.max_tokens,
            "single_runs": args.single_runs,
            "requests_per_concurrency": args.requests,
        },
        "single_request": single_result,
        "concurrent": concurrent_results,
    }

    if args.output == "json":
        print(json.dumps(output, indent=2))
    else:
        _print_table(output)

    return output


def _print_table(result):
    s = result["single_request"]
    print(f"\nModel   : {result['model']} ({result['engine']})")
    print(f"TTFT    : {s['ttft_ms']} ms")
    print(f"Tok/s   : {s['tokens_per_sec']}")
    print(f"E2E     : {s['e2e_latency_ms']} ms")
    print(f"\nConcurrent throughput:")
    for c, r in result["concurrent"].items():
        if "throughput_tokens_per_sec" in r:
            print(f"  @{c:>2}: {r['throughput_tokens_per_sec']:>8.1f} tok/s  "
                  f"p50={r['p50_e2e_ms']:.0f}ms  p95={r['p95_e2e_ms']:.0f}ms")
        else:
            print(f"  @{c:>2}: ERROR — {r.get('error')}")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
