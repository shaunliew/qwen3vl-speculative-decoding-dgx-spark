"""
run_client.py: HTTP benchmark client for vllm serve.

This script is the counterpart to start_server.sh. While the server (Terminal 1)
keeps the model loaded and the speculative config active, this client (Terminal 2)
sends benchmark questions one by one and measures timing.

WHY HTTP CLIENT INSTEAD OF vLLM PYTHON API?
  The vllm serve + HTTP pattern separates model loading from benchmarking.
  You can restart the client without reloading the 60 GB model.
  The OpenAI-compatible REST API is also what production apps use, this demo
  is realistic rather than a one-off test harness.

Usage (inside the container, after start_server.sh is running):
  python src/run_client.py --method baseline
  python src/run_client.py --method ngram
  python src/run_client.py --method draft_target
  python src/run_client.py --method eagle3

Acceptance rate: visible in Terminal 1 (server logs). Search for "spec_decode".
"""

import argparse
import json
import sys
import time

import requests

# ── make src/ imports work regardless of working directory ───────────────────
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from benchmark_harness import (
    NUM_WARMUP,
    ExperimentResult,
    SampleResult,
    extract_answer_letter,
    get_gpu_memory_gb,
    save_results,
)
from question_loader import load_formatted


# ── Server health check ───────────────────────────────────────────────────────

def wait_for_server(endpoint, timeout=300):
    """
    Poll GET /health every 3 seconds until the server responds 200.

    vllm serve takes 30–120 seconds to load a 60 GB model and capture CUDA graphs.
    This function blocks until the server is ready, so we do not need to manually
    time the model load or guess when to start the client.

    Raises RuntimeError if the server does not respond within `timeout` seconds.
    """
    health_url = f"{endpoint}/health"
    deadline = time.time() + timeout
    attempt = 0

    print(f"Waiting for server at {health_url} (timeout={timeout}s)...")
    while time.time() < deadline:
        attempt += 1
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                print(f"  Server ready after ~{attempt * 3}s")
                return
        except (ConnectionRefusedError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout):
            pass  # Server not up yet, normal during model load
        time.sleep(3)

    raise RuntimeError(
        f"Server at {endpoint} did not become healthy within {timeout}s. "
        "Check Terminal 1 for errors (OOM, missing model files, wrong CUDA version)."
    )


def get_model_name(endpoint):
    """
    Query GET /v1/models and return the first model id.

    vllm serve registers the loaded model under its HuggingFace repo id.
    We need this string to fill in the 'model' field of chat completion requests.
    The alternative (hardcoding the model name here) would break if the server
    was started with a different model, dynamically fetching it keeps things robust.
    """
    resp = requests.get(f"{endpoint}/v1/models", timeout=10)
    resp.raise_for_status()
    models = resp.json()["data"]
    if not models:
        raise RuntimeError("No models found on server, did vllm serve start correctly?")
    return models[0]["id"]


# ── Single-sample inference via streaming ─────────────────────────────────────

def run_one_sample(endpoint, model_name, messages):
    """
    Send one chat completion request and parse the streaming response.

    WHY stream=True?
      Streaming (Server-Sent Events / SSE) gives us token-by-token deltas.
      The very first delta with non-empty content marks the end of prefill —
      that timestamp is TTFT. Without streaming we would only get the wall time.

    WHY include_usage in stream_options?
      The final SSE event carries usage.completion_tokens, the total number of
      tokens generated. We need this to compute TPOT = decode_ms / (tokens - 1).
      Without this flag, usage is omitted from streaming responses.

    Returns: (output_text, num_tokens, ttft_ms, wall_ms)
      ttft_ms , None if no content delta arrived (e.g. empty response)
      wall_ms , total elapsed time including network round-trip
    """
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        # temperature=0 + seed=42: fully deterministic output.
        # Essential so baseline vs speculative answers are identical, this
        # proves that speculative decoding does not change model outputs.
        "temperature": 0,
        "seed": 42,
        # max_tokens=1000: allows 200-400 token explanations with safety headroom.
        # AngelSlim benchmarks Eagle3 at 1024 tokens. With 1000 tokens per sample
        # and 17 samples, we get ~8500 total output tokens and ~1700 speculation
        # rounds minimum, enough to measure acceptance rate within plus or minus 2-3%
        # and distinguish real speedup from noise.
        # Thinking mode is disabled server-side via --default-chat-template-kwargs in
        # start_server.sh, so no per-request override is needed here.
        "max_tokens": 1000,
        # Streaming gives us TTFT by detecting the first content delta
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    output_text = ""
    num_tokens = 0
    first_token_time = None  # set when the first non-empty delta arrives

    t_start = time.perf_counter()

    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue  # SSE uses blank lines as keepalives, ignore them

            # Each SSE line starts with "data: ", strip that prefix
            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]

            # The stream ends with a sentinel "data: [DONE]", nothing to parse
            if data_str.strip() == "[DONE]":
                break

            chunk = json.loads(data_str)

            # ── Detect first token (end of prefill) ────────────────────────
            # Each chunk has choices[0].delta.content, it is empty string ""
            # during prefill setup, then non-empty once the first token is ready.
            choices = chunk.get("choices", [])
            if choices:
                delta_content = choices[0].get("delta", {}).get("content", "")
                if delta_content and first_token_time is None:
                    # This is the moment the first output token arrived —
                    # everything before this was prefill (image encoding, KV fill).
                    first_token_time = time.perf_counter()
                output_text += delta_content or ""

            # ── Extract token count from final usage event ─────────────────
            # The last chunk (before [DONE]) has usage.completion_tokens.
            # This is the authoritative token count, do not count deltas manually
            # because some chunks carry multiple tokens silently.
            usage = chunk.get("usage")
            if usage:
                num_tokens = usage.get("completion_tokens", num_tokens)

    t_end = time.perf_counter()
    wall_ms = (t_end - t_start) * 1000
    ttft_ms = (first_token_time - t_start) * 1000 if first_token_time else None

    return output_text, num_tokens, ttft_ms, wall_ms


# ── Full benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(method, endpoint):
    """
    Send all benchmark questions to the server and collect timing metrics.

    The first NUM_WARMUP samples are sent but not recorded. vLLM may still be
    capturing CUDA graphs on the first few requests even after /health returns 200,
    so warmup requests flush that overhead before we start measuring.
    """
    model_name = get_model_name(endpoint)
    print(f"Model: {model_name}")

    formatted_samples = load_formatted()
    total = len(formatted_samples)
    result = ExperimentResult(method=method)
    peak_memory_captured = False

    for i, (messages, expected_answer) in enumerate(formatted_samples):
        is_warmup = i < NUM_WARMUP

        if is_warmup:
            print(f"  Warmup {i + 1}/{NUM_WARMUP}...", end=" ", flush=True)
        else:
            measured_i = i - NUM_WARMUP + 1
            print(f"  Sample {measured_i}/{total - NUM_WARMUP}...", end=" ", flush=True)

        output_text, num_tokens, ttft_ms, wall_ms = run_one_sample(
            endpoint, model_name, messages
        )

        # ── Compute decode TPOT ────────────────────────────────────────────
        # TPOT = time per output token, decode phase only (excludes prefill).
        # Token 1 is produced at the end of prefill (that cost is in TTFT).
        # Tokens 2..N are the decode tokens that speculative decoding speeds up.
        # Formula: (wall_ms - ttft_ms) / (num_tokens - 1)
        decode_tokens = max(num_tokens - 1, 1)
        if ttft_ms is not None and num_tokens > 1:
            decode_ms = wall_ms - ttft_ms
            decode_tpot_ms = decode_ms / decode_tokens
            tokens_per_sec = decode_tokens / (decode_ms / 1000)
        else:
            # Fallback: treat entire wall time as decode (underestimates TTFT cost)
            decode_tpot_ms = wall_ms / decode_tokens
            tokens_per_sec = None

        predicted = extract_answer_letter(output_text)
        correct = predicted == expected_answer.upper()

        # ── Live progress line ─────────────────────────────────────────────
        ttft_str = f"TTFT={ttft_ms:.0f}ms" if ttft_ms is not None else "TTFT=N/A"
        tpot_str = f"TPOT={decode_tpot_ms:.1f}ms" if decode_tpot_ms is not None else "TPOT=N/A"
        mark = "✓" if correct else "✗"
        print(f"{wall_ms:.0f}ms wall | {ttft_str} | {tpot_str} | {num_tokens} tok → {predicted} ({mark})")

        if not is_warmup:
            # ── Capture peak memory once after warmup ──────────────────────
            # We read GPU memory here (not during warmup) so the KV cache and
            # any draft model weights are already fully allocated. nvidia-smi
            # sees all GPU processes including the vllm serve EngineCore child.
            if not peak_memory_captured:
                result.peak_gpu_memory_gb = get_gpu_memory_gb()
                peak_memory_captured = True

            # spec_acceptance_rate is not available per-request via the HTTP API.
            # The true per-request rate is visible in Terminal 1 server logs.
            # Search for "spec_decode_acceptance_rate" in those logs.
            result.samples.append(SampleResult(
                sample_id=i,
                wall_time_ms=wall_ms,
                ttft_ms=ttft_ms,
                decode_tpot_ms=decode_tpot_ms,
                tokens_per_sec=tokens_per_sec,
                num_output_tokens=num_tokens,
                spec_acceptance_rate=None,  # see server logs (Terminal 1)
                predicted=predicted,
                expected=expected_answer,
                correct=correct,
            ))

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HTTP benchmark client for vllm serve speculative decoding demo."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["baseline", "ngram", "draft_target", "eagle3"],
        help="Which speculative decoding method the server was started with.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="Base URL of the running vllm serve instance (default: http://localhost:8000).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Speculative Decoding Benchmark, {args.method.upper()}")
    print(f"Endpoint : {args.endpoint}")
    print(f"Warmup   : {NUM_WARMUP} samples (excluded from results)")
    print("=" * 60)
    print()

    # Block here until the server is ready, model load takes 30–120 seconds
    wait_for_server(args.endpoint)

    result = run_benchmark(args.method, args.endpoint)
    out_path = save_results(result)

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("─" * 60)
    print(f"METHOD          : {args.method}")
    print(f"Samples         : {len(result.samples)}")
    if result.mean_ttft_ms is not None:
        print(f"Mean TTFT       : {result.mean_ttft_ms:.0f} ms  (prefill, same across all methods)")
    if result.mean_decode_tpot_ms is not None:
        print(f"Mean decode TPOT: {result.mean_decode_tpot_ms:.1f} ms/tok  ← lower = faster speculation")
    print(f"Peak GPU memory : {result.peak_gpu_memory_gb:.1f} GB")
    print(f"Accuracy        : {result.accuracy * 100:.1f}%  (should be ~same across all methods)")
    print(f"Results saved   : {out_path}")
    print()
    print("Acceptance rate → check Terminal 1 (server logs), search 'spec_decode'.")
    print("Compare methods → python src/benchmark_compare.py")
    print("─" * 60)


if __name__ == "__main__":
    main()
