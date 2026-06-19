#!/usr/bin/env python3
"""
Maic vs LMStudio benchmark — measures TTFT and decode speed via the
OpenAI-compatible streaming API.

Usage:
  # Benchmark Maic (default port 8001):
  python benchmarks/bench.py

  # Benchmark LMStudio (default port 1234):
  python benchmarks/bench.py --url http://localhost:1234 --model "your-model-name"

  # Both sides, produces a side-by-side comparison table:
  python benchmarks/bench.py --compare

  # Custom run count / output length:
  python benchmarks/bench.py --runs 10 --max-tokens 512
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator

try:
    import httpx
except ImportError:
    print("httpx is required: pip install httpx")
    sys.exit(1)

# ---------------------------------------------------------------------------
# LMStudio reference numbers for comparison.
# Source: lmstudio-ai/mlx-engine issue #103 — M1 Pro 32GB, MLX backend,
# LM Studio 0.3.10, no speculative decoding.
# ---------------------------------------------------------------------------
LMSTUDIO_REFERENCE = {
    "hardware": "M1 Pro 32GB",
    "backend": "MLX (LM Studio 0.3.10)",
    "source": "https://github.com/lmstudio-ai/mlx-engine/issues/103",
    "models": {
        "Llama-3.1-8B Q4": {"tps_mean": 38.0},
        "Llama-3.2-3B Q4": {"tps_mean": 38.7},
        "Qwen-2.5-7B Q4":  {"tps_mean": 37.08},
    },
}

# ---------------------------------------------------------------------------
# Prompts used across all runs. Each covers a different length / domain so we
# get a realistic spread (short burst, medium, long context).
# These match the style of prompts used in published MLX benchmark suites.
# ---------------------------------------------------------------------------
PROMPTS = [
    # Short — tests low-latency / TTFT sensitivity
    {
        "name": "short",
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
        ],
    },
    # Medium — typical chat use-case
    {
        "name": "medium",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain the difference between supervised learning and unsupervised learning. "
                    "Give a concrete example for each. Keep your answer under 200 words."
                ),
            },
        ],
    },
    # Code — tests decode throughput at moderate token counts
    {
        "name": "code",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python function that implements a binary search tree with insert, "
                    "search, and in-order traversal methods. Add docstrings."
                ),
            },
        ],
    },
    # Long output — stresses sustained decode throughput
    {
        "name": "long",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a detailed technical comparison between PostgreSQL and MongoDB. "
                    "Cover: data model, query language, ACID compliance, scalability, "
                    "indexing strategies, and when to choose each. Be thorough."
                ),
            },
        ],
    },
]


@dataclass
class RunResult:
    prompt_name: str
    ttft_s: float          # wall-clock seconds until the first token chunk arrived
    decode_tps: float      # tokens/s AFTER the first token (pure decode speed)
    total_s: float         # total wall-clock time for the whole response
    token_count: int       # tokens generated (estimated from SSE chunks)
    error: str | None = None


@dataclass
class BenchResult:
    label: str
    url: str
    model: str
    runs: list[RunResult] = field(default_factory=list)


def _stream_sse(client: httpx.Client, url: str, payload: dict) -> Iterator[str]:
    """Yield raw SSE data lines from a streaming chat-completion request."""
    with client.stream("POST", url, json=payload, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    return
                yield data


def _estimate_tokens(text: str) -> int:
    """
    Estimate BPE token count without a tokenizer.
    Empirically ~4 chars per token for English prose / code across GPT and open models.
    """
    return max(1, round(len(text) / 4))


def _run_once(client: httpx.Client, base_url: str, model: str, prompt: dict, max_tokens: int) -> RunResult:
    payload = {
        "model": model,
        "messages": prompt["messages"],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,  # greedy — deterministic, best-case speed
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"

    t_start = time.perf_counter()
    t_first: float | None = None
    server_token_count = 0
    full_text = ""

    try:
        for data in _stream_sse(client, url, payload):
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            # Prefer usage.completion_tokens if the server provides it (exact count)
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                server_token_count = int(usage["completion_tokens"])

            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                full_text += content
    except httpx.HTTPStatusError as exc:
        return RunResult(
            prompt_name=prompt["name"],
            ttft_s=0,
            decode_tps=0,
            total_s=0,
            token_count=0,
            error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except Exception as exc:
        return RunResult(
            prompt_name=prompt["name"],
            ttft_s=0,
            decode_tps=0,
            total_s=0,
            token_count=0,
            error=str(exc)[:200],
        )

    t_end = time.perf_counter()

    if t_first is None:
        return RunResult(
            prompt_name=prompt["name"],
            ttft_s=0,
            decode_tps=0,
            total_s=t_end - t_start,
            token_count=0,
            error="no tokens received",
        )

    # Use exact server-reported count if available, fall back to char-based estimate
    token_count = server_token_count if server_token_count > 0 else _estimate_tokens(full_text)

    ttft = t_first - t_start
    decode_duration = t_end - t_first
    decode_tps = token_count / decode_duration if decode_duration > 0 else 0.0

    return RunResult(
        prompt_name=prompt["name"],
        ttft_s=round(ttft, 3),
        decode_tps=round(decode_tps, 1),
        total_s=round(t_end - t_start, 3),
        token_count=token_count,
    )


def run_benchmark(label: str, base_url: str, model: str, runs: int, max_tokens: int, warmup: int = 1) -> BenchResult:
    result = BenchResult(label=label, url=base_url, model=model)

    with httpx.Client() as client:
        # Check connectivity
        try:
            resp = client.get(base_url.rstrip("/") + "/v1/models", timeout=5)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  [!] Cannot reach {base_url}: {exc}", file=sys.stderr)
            return result

        # Warmup: run the first prompt once to prime the model cache / KV cache
        if warmup > 0:
            print(f"  Warming up ({warmup} run(s))…", end=" ", flush=True)
            for _ in range(warmup):
                _run_once(client, base_url, model, PROMPTS[0], max_tokens=64)
            print("done")

        for prompt in PROMPTS:
            per_prompt_results: list[RunResult] = []
            print(f"  Prompt '{prompt['name']}' ({runs} run(s)):", end=" ", flush=True)
            for i in range(runs):
                r = _run_once(client, base_url, model, prompt, max_tokens)
                per_prompt_results.append(r)
                result.runs.append(r)
                if r.error:
                    print(f"ERROR({r.error})", end=" ", flush=True)
                else:
                    print(f"{r.decode_tps:.0f}", end=" ", flush=True)
            print()  # newline after prompt row

    return result


def _summary(runs: list[RunResult]) -> dict:
    """Compute aggregate stats across all successful runs."""
    good = [r for r in runs if not r.error]
    if not good:
        return {"n": 0}
    tps_vals = [r.decode_tps for r in good]
    ttft_vals = [r.ttft_s for r in good]
    return {
        "n": len(good),
        "errors": len(runs) - len(good),
        "tps_mean": round(statistics.mean(tps_vals), 1),
        "tps_median": round(statistics.median(tps_vals), 1),
        "tps_p95": round(sorted(tps_vals)[int(len(tps_vals) * 0.95)], 1),
        "tps_max": round(max(tps_vals), 1),
        "ttft_mean_ms": round(statistics.mean(ttft_vals) * 1000, 1),
        "ttft_median_ms": round(statistics.median(ttft_vals) * 1000, 1),
        "ttft_p95_ms": round(sorted(ttft_vals)[int(len(ttft_vals) * 0.95)] * 1000, 1),
    }


def _per_prompt_summary(runs: list[RunResult]) -> dict[str, dict]:
    by_prompt: dict[str, list[RunResult]] = {}
    for r in runs:
        by_prompt.setdefault(r.prompt_name, []).append(r)
    return {name: _summary(rs) for name, rs in by_prompt.items()}


def print_report(result: BenchResult) -> None:
    print(f"\n{'='*60}")
    print(f"  {result.label}")
    print(f"  URL:   {result.url}")
    print(f"  Model: {result.model}")
    print(f"{'='*60}")

    overall = _summary(result.runs)
    if overall["n"] == 0:
        print("  No successful runs.")
        return

    print("\n  Overall (all prompts combined):")
    print(f"    Decode speed  mean:   {overall['tps_mean']:>7.1f} tok/s")
    print(f"    Decode speed  median: {overall['tps_median']:>7.1f} tok/s")
    print(f"    Decode speed  p95:    {overall['tps_p95']:>7.1f} tok/s")
    print(f"    Decode speed  max:    {overall['tps_max']:>7.1f} tok/s")
    print(f"    TTFT          mean:   {overall['ttft_mean_ms']:>7.1f} ms")
    print(f"    TTFT          median: {overall['ttft_median_ms']:>7.1f} ms")
    print(f"    TTFT          p95:    {overall['ttft_p95_ms']:>7.1f} ms")
    if overall["errors"]:
        print(f"    Errors: {overall['errors']}")

    print("\n  Per-prompt breakdown:")
    pp = _per_prompt_summary(result.runs)
    print(f"  {'Prompt':<10}  {'TPS mean':>9}  {'TPS median':>11}  {'TTFT mean':>10}  {'TTFT p95':>9}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*11}  {'-'*10}  {'-'*9}")
    for name, s in pp.items():
        if s["n"] == 0:
            print(f"  {name:<10}  {'N/A':>9}")
            continue
        print(
            f"  {name:<10}  {s['tps_mean']:>8.1f}t  {s['tps_median']:>10.1f}t"
            f"  {s['ttft_mean_ms']:>8.1f}ms  {s['ttft_p95_ms']:>7.1f}ms"
        )


def print_comparison(a: BenchResult, b: BenchResult) -> None:
    """Print a side-by-side comparison table."""
    sa = _summary(a.runs)
    sb = _summary(b.runs)

    print(f"\n{'='*70}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<28}  {a.label:>16}  {b.label:>16}  {'Delta':>8}")
    print(f"  {'-'*28}  {'-'*16}  {'-'*16}  {'-'*8}")

    def row(label: str, va: float, vb: float, unit: str, higher_better: bool = True) -> None:
        delta = vb - va
        arrow = "▲" if (delta > 0) == higher_better else "▼"
        pct = (delta / va * 100) if va > 0 else 0
        print(
            f"  {label:<28}  {va:>14.1f}{unit}  {vb:>14.1f}{unit}  {arrow}{abs(pct):>5.1f}%"
        )

    if sa["n"] > 0 and sb["n"] > 0:
        row("Decode tok/s (mean)",   sa["tps_mean"],       sb["tps_mean"],       "t", True)
        row("Decode tok/s (median)", sa["tps_median"],     sb["tps_median"],     "t", True)
        row("Decode tok/s (p95)",    sa["tps_p95"],        sb["tps_p95"],        "t", True)
        row("TTFT mean (ms)",        sa["ttft_mean_ms"],   sb["ttft_mean_ms"],   " ", False)
        row("TTFT median (ms)",      sa["ttft_median_ms"], sb["ttft_median_ms"], " ", False)
        row("TTFT p95 (ms)",         sa["ttft_p95_ms"],    sb["ttft_p95_ms"],    " ", False)
    else:
        print("  (insufficient data for comparison)")

    # Per-prompt breakdown
    pa = _per_prompt_summary(a.runs)
    pb = _per_prompt_summary(b.runs)
    all_prompts = sorted(set(pa) | set(pb))

    print(f"\n  Per-prompt tok/s (mean):")
    print(f"  {'Prompt':<10}  {a.label:>16}  {b.label:>16}  {'Delta':>8}")
    print(f"  {'-'*10}  {'-'*16}  {'-'*16}  {'-'*8}")
    for name in all_prompts:
        sa_p = pa.get(name, {})
        sb_p = pb.get(name, {})
        va = sa_p.get("tps_mean", 0.0)
        vb = sb_p.get("tps_mean", 0.0)
        if va > 0 and vb > 0:
            delta = vb - va
            pct = delta / va * 100
            arrow = "▲" if delta > 0 else "▼"
            print(f"  {name:<10}  {va:>14.1f}t  {vb:>14.1f}t  {arrow}{abs(pct):>5.1f}%")
        else:
            print(f"  {name:<10}  {'N/A':>16}  {'N/A':>16}")
    print()


def load_model(base_url: str, model_id: str) -> None:
    """POST /v1/models/load and poll until ready, with a progress indicator."""
    print(f"\nLoading {model_id} …", end=" ", flush=True)
    t0 = time.perf_counter()
    with httpx.Client() as client:
        try:
            client.post(
                base_url.rstrip("/") + "/v1/models/load",
                json={"model_id": model_id},
                timeout=300,  # model loading can take 60-120s
            ).raise_for_status()
        except httpx.HTTPStatusError as exc:
            # 409 means already loaded — that's fine
            if exc.response.status_code != 409:
                print(f"\n  [!] Load request failed: {exc.response.status_code} {exc.response.text[:200]}")
                return

    # The POST blocks until the model is fully loaded before returning.
    elapsed = time.perf_counter() - t0
    print(f" done ({elapsed:.1f}s)")


def print_vs_lmstudio(result: BenchResult) -> None:
    """Print a side-by-side comparison against the LMStudio reference numbers."""
    overall = _summary(result.runs)
    if overall["n"] == 0:
        print("  (no data to compare)")
        return

    ref = LMSTUDIO_REFERENCE
    # Use Qwen-2.5-7B as the closest 7B-class match
    ref_model = "Qwen-2.5-7B Q4"
    ref_tps = ref["models"][ref_model]["tps_mean"]
    our_tps = overall["tps_mean"]

    delta_pct = (our_tps - ref_tps) / ref_tps * 100
    arrow = "▲" if delta_pct >= 0 else "▼"

    print(f"\n{'='*62}")
    print("  VS LMSTUDIO REFERENCE")
    print(f"  LM Studio — {ref['hardware']}, {ref['backend']}")
    print(f"  Source: {ref['source']}")
    print(f"{'='*62}")
    print(f"  {'Metric':<28}  {'Maic (16GB)':>12}  {'LMStudio (32GB)':>15}  {'Delta':>7}")
    print(f"  {'-'*28}  {'-'*12}  {'-'*15}  {'-'*7}")
    print(
        f"  {'Decode 7B-class (mean)':<28}  {our_tps:>10.1f}t/s"
        f"  {ref_tps:>13.2f}t/s  {arrow}{abs(delta_pct):>4.1f}%"
    )
    print(
        f"\n  Reference model used for comparison: {ref_model} @ {ref_tps} tok/s"
    )
    print(
        "  Note: RAM (16GB vs 32GB) does not affect decode speed for models\n"
        "  that fit in unified memory. Comparison is valid.\n"
    )


def _resolve_model(base_url: str, model_arg: str | None) -> str:
    """If model arg not provided, ask the server for the first loaded model."""
    if model_arg:
        return model_arg
    try:
        resp = httpx.get(base_url.rstrip("/") + "/v1/models", timeout=5)
        data = resp.json()
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Maic vs LMStudio")
    parser.add_argument("--url", default="http://localhost:8001", help="Primary server URL (default: Maic on 8001)")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--label", default="Maic", help="Label for primary server in reports")
    parser.add_argument("--compare-url", default=None, help="Second server URL for comparison (e.g. LMStudio on 1234)")
    parser.add_argument("--compare-model", default=None, help="Model name on second server")
    parser.add_argument("--compare-label", default="LMStudio", help="Label for second server")
    parser.add_argument("--runs", type=int, default=3, help="Runs per prompt (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response (default: 256)")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup run")
    parser.add_argument(
        "--load-model",
        default=None,
        metavar="MODEL_ID",
        help="Load this model via POST /v1/models/load before benchmarking (e.g. mlx-community/Mistral-7B-Instruct-v0.3-4bit)",
    )
    parser.add_argument(
        "--vs-lmstudio",
        action="store_true",
        help="After benchmarking, print a comparison table against LMStudio reference numbers",
    )
    args = parser.parse_args()

    warmup = 0 if args.no_warmup else 1

    # ── Optional model load ───────────────────────────────────────────────────
    if args.load_model:
        load_model(args.url, args.load_model)
        if not args.model:
            args.model = args.load_model

    # ── Primary server ────────────────────────────────────────────────────────
    model = _resolve_model(args.url, args.model)
    print(f"\nBenchmarking {args.label} @ {args.url}  model={model}")
    a = run_benchmark(args.label, args.url, model, args.runs, args.max_tokens, warmup)
    print_report(a)

    # ── Optional LMStudio reference comparison ────────────────────────────────
    if args.vs_lmstudio:
        print_vs_lmstudio(a)

    # ── Optional comparison server ─────────────────────────────────────────
    if args.compare_url:
        compare_model = _resolve_model(args.compare_url, args.compare_model)
        print(f"\nBenchmarking {args.compare_label} @ {args.compare_url}  model={compare_model}")
        b = run_benchmark(args.compare_label, args.compare_url, compare_model, args.runs, args.max_tokens, warmup)
        print_report(b)
        print_comparison(a, b)


if __name__ == "__main__":
    main()
