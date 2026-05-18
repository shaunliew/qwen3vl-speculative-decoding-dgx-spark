"""
benchmark_harness.py: Data models and IO utilities for experiment results.

This module is a pure data model + IO layer. Timing is handled in run_client.py
via SSE streaming against a running `vllm serve` HTTP server, not here.

run_client.py sends requests to the vLLM OpenAI-compatible endpoint, reads the
Server-Sent Events stream to measure TTFT and per-token decode timing, then
constructs SampleResult / ExperimentResult objects and calls save_results().

KEY METRICS EXPLAINED:

  TTFT (Time To First Token):
    How long from request start until the first output token appears.
    For VLMs this is mostly prefill, the vision encoder processing image patches.
    Speculative decoding does NOT reduce TTFT. Expect 0.5–1 seconds for text-only prompts.

  Decode TPOT (Time Per Output Token, decode-only):
    Milliseconds per token AFTER the first token. This is the decode phase —
    the token-by-token loop that speculative decoding accelerates.
    Formula: (finished_time - first_token_time) / (num_tokens - 1)
    Lower is better.

  Speedup:
    baseline_decode_tpot / method_decode_tpot
    A speedup of 2.0x means decode is twice as fast.

  Acceptance rate:
    Fraction of draft tokens the target model accepted without resampling.
    Only meaningful for speculative methods (ngram, draft-target, eagle3).
    Higher acceptance → more free tokens → bigger speedup.

  Accuracy:
    Fraction of questions answered correctly.
    Speculative decoding is lossless, this should be identical across all methods.
"""

import json
import re
import subprocess
from pathlib import Path


# Where we save results, one JSON file per method
RESULTS_DIR = Path(__file__).parent.parent / "results"


def get_gpu_memory_gb():
    """
    Return current GPU memory usage in GB by querying nvidia-smi.

    WHY NOT torch.cuda.max_memory_reserved()?
    vLLM loads the model in a child subprocess (EngineCore). The main Python
    process never allocates GPU memory itself, so torch.cuda always returns 0.
    nvidia-smi queries the GPU driver directly and sees all processes.

    WHY TWO QUERIES?
    The DGX Spark uses a GB10 Grace Blackwell chip with UNIFIED memory (CPU and
    GPU share the same LPDDR5X pool). On unified memory hardware, the standard
    --query-gpu=memory.used returns [N/A] because there is no separate VRAM.
    The --query-compute-apps=used_memory query instead reports memory consumed
    by each running CUDA application, which DOES work on unified memory.
    We try the standard query first (works on discrete GPUs like A100/H100),
    then fall back to the compute-apps query for GB10 / unified memory systems.
    """
    # Try 1: standard discrete-GPU query (A100, H100, RTX etc.)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        )
        raw = out.strip().split("\n")[0].strip()
        if raw and raw.lower() not in ("[n/a]", "n/a", ""):
            return round(float(raw) / 1024, 1)
    except Exception:
        pass

    # Try 2: compute-apps query, works on GB10 unified memory (DGX Spark)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        )
        # Sum MiB across all compute processes (vLLM EngineCore + APIServer)
        lines = [l.strip() for l in out.strip().split("\n")
                 if l.strip() and l.strip().lower() not in ("[n/a]", "n/a", "")]
        if lines:
            total_mib = sum(float(l) for l in lines)
            return round(total_mib / 1024, 1)
    except Exception:
        pass

    return 0.0
RESULTS_DIR.mkdir(exist_ok=True)

# Discard this many requests at the start, vLLM captures CUDA graphs on the
# first few forward passes which makes them much slower than steady state.
NUM_WARMUP = 3


class SampleResult:
    """All metrics for one benchmark question."""

    def __init__(self, sample_id, wall_time_ms, ttft_ms, decode_tpot_ms,
                 tokens_per_sec, num_output_tokens, spec_acceptance_rate,
                 predicted, expected, correct):
        self.sample_id          = sample_id           # index of this sample
        self.wall_time_ms       = wall_time_ms        # total time (prefill + decode)
        self.ttft_ms            = ttft_ms             # time to first token; None if unavailable
        self.decode_tpot_ms     = decode_tpot_ms      # decode-only ms/token; None if unavailable
        self.tokens_per_sec     = tokens_per_sec      # decode throughput; None if unavailable
        self.num_output_tokens  = num_output_tokens   # total tokens generated
        self.spec_acceptance_rate = spec_acceptance_rate  # None for non-speculative methods
        self.predicted          = predicted           # model's answer letter (A/B/C/D)
        self.expected           = expected            # correct answer letter
        self.correct            = correct             # True if predicted == expected


class ExperimentResult:
    """Aggregated metrics for one full experiment run."""

    def __init__(self, method):
        self.method             = method   # "baseline", "ngram", "draft_target", or "eagle3"
        self.peak_gpu_memory_gb = 0.0      # set after LLM() loads, in GB
        self.samples            = []       # list of SampleResult (warmup excluded)

    # ── simple percentile helper ───────────────────────────────────────────
    def _pct(self, values, p):
        """Return the p-th percentile of a list, ignoring None values."""
        valid = sorted(v for v in values if v is not None)
        if not valid:
            return None
        return valid[int(len(valid) * p)]

    # ── TTFT aggregates ────────────────────────────────────────────────────
    @property
    def mean_ttft_ms(self):
        values = [s.ttft_ms for s in self.samples if s.ttft_ms is not None]
        return sum(values) / len(values) if values else None

    @property
    def median_ttft_ms(self):
        return self._pct([s.ttft_ms for s in self.samples], 0.5)

    @property
    def p95_ttft_ms(self):
        return self._pct([s.ttft_ms for s in self.samples], 0.95)

    # ── Decode TPOT aggregates ─────────────────────────────────────────────
    @property
    def mean_decode_tpot_ms(self):
        values = [s.decode_tpot_ms for s in self.samples if s.decode_tpot_ms is not None]
        return sum(values) / len(values) if values else None

    @property
    def mean_tpot_ms(self):
        # Alias kept so benchmark_compare.py works with both old and new JSON files
        return self.mean_decode_tpot_ms

    @property
    def median_decode_tpot_ms(self):
        return self._pct([s.decode_tpot_ms for s in self.samples], 0.5)

    @property
    def p95_decode_tpot_ms(self):
        # p95 shows the slow tail, important for speculative decoding because
        # acceptance rate varies per sample (low on complex reasoning, high on simple answers)
        return self._pct([s.decode_tpot_ms for s in self.samples], 0.95)

    # ── Throughput and acceptance ──────────────────────────────────────────
    @property
    def mean_tokens_per_sec(self):
        values = [s.tokens_per_sec for s in self.samples if s.tokens_per_sec is not None]
        return sum(values) / len(values) if values else None

    @property
    def mean_spec_acceptance_rate(self):
        values = [s.spec_acceptance_rate for s in self.samples if s.spec_acceptance_rate is not None]
        return sum(values) / len(values) if values else None

    # ── Simple aggregates ──────────────────────────────────────────────────
    @property
    def total_tokens(self):
        return sum(s.num_output_tokens for s in self.samples)

    @property
    def accuracy(self):
        if not self.samples:
            return 0.0
        return sum(s.correct for s in self.samples) / len(self.samples)

    def to_dict(self):
        """Serialize to a plain dict for JSON saving."""
        def r(v, n=2):
            # Round v to n decimal places, or return None if v is None
            return round(v, n) if v is not None else None

        return {
            "method": self.method,
            # Decode TPOT, the primary speculative decoding metric
            "mean_decode_tpot_ms":   r(self.mean_decode_tpot_ms),
            "median_decode_tpot_ms": r(self.median_decode_tpot_ms),
            "p95_decode_tpot_ms":    r(self.p95_decode_tpot_ms),
            # Backward-compat alias for older result files
            "mean_tpot_ms":          r(self.mean_decode_tpot_ms),
            # TTFT, should be the same across all methods (sanity check)
            "mean_ttft_ms":          r(self.mean_ttft_ms),
            "median_ttft_ms":        r(self.median_ttft_ms),
            "p95_ttft_ms":           r(self.p95_ttft_ms),
            # Throughput and speculation quality
            "mean_tokens_per_sec":        r(self.mean_tokens_per_sec),
            "mean_spec_acceptance_rate":  r(self.mean_spec_acceptance_rate, 3),
            # Hardware and accuracy
            "peak_gpu_memory_gb": r(self.peak_gpu_memory_gb),
            "total_tokens":       self.total_tokens,
            "accuracy":           r(self.accuracy, 3),
            "num_samples":        len(self.samples),
            # Per-sample breakdown
            "samples": [
                {
                    "id":                   s.sample_id,
                    "wall_time_ms":         r(s.wall_time_ms),
                    "ttft_ms":              r(s.ttft_ms),
                    "decode_tpot_ms":       r(s.decode_tpot_ms),
                    "tpot_ms":              r(s.decode_tpot_ms),   # alias
                    "tokens_per_sec":       r(s.tokens_per_sec),
                    "num_tokens":           s.num_output_tokens,
                    "spec_acceptance_rate": r(s.spec_acceptance_rate, 3),
                    "predicted":            s.predicted,
                    "expected":             s.expected,
                    "correct":              s.correct,
                }
                for s in self.samples
            ],
        }


def extract_answer_letter(text):
    """
    Pull the answer letter (A/B/C/D) from the model's response.

    With enable_thinking=False set in chat_template_kwargs, Qwen3 answers
    directly without a <think> block, so the first line is typically the
    answer letter followed by the explanation.
    """
    text = text.strip()

    first_line = text.split("\n")[0].strip()
    if first_line.upper() in ("A", "B", "C", "D", "E"):
        return first_line.upper()

    match = re.search(r"\b([ABCDE])\b", text.upper())
    if match:
        return match.group(1)

    return text[0].upper() if text else "?"



def save_results(result):
    """Save experiment results to results/<method>.json."""
    out_path = RESULTS_DIR / f"{result.method}.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"\nResults saved → {out_path}")
    return out_path


def load_results(method):
    """Load previously saved results for a method. Returns None if not found."""
    path = RESULTS_DIR / f"{method}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
