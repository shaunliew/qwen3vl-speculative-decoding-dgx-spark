"""
benchmark_compare.py: Load all experiment results and print a comparison table.

HOW TO READ THE TABLE:
  TPOT (Time Per Output Token): milliseconds to generate each decode token.
    Lower is better. Speculative decoding only affects this metric, not TTFT.

  Speedup: TPOT_baseline / TPOT_this_method.
    1.0x = same as baseline.  2.0x = twice as fast.
    Anything above ~1.3x is considered meaningful for a VLM workload.

  Acceptance Rate: fraction of draft tokens the target model kept without
    resampling.  Higher is better.  ~0.7+ is a good sign for a well-matched
    draft model.  N/gram results do not report this, only draft-model methods do.

  Accuracy: fraction of questions answered correctly.
    Speculative decoding is mathematically lossless, so this should stay
    within noise of the baseline.  Any large drop indicates a bug.

Usage:
  python src/benchmark_compare.py
"""

import sys
from pathlib import Path

# Allow running directly from the project root
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_harness import load_results  # noqa: E402

# Ordered list of (display_name, method_key), method_key matches results/<key>.json
METHODS = [
    ("Baseline (no speculation)", "baseline"),
    ("N-gram speculation",        "ngram"),
    ("Draft-target (8B→32B)", "draft_target"),
    ("Eagle3 (AngelSlim head)",   "eagle3"),
]


def _fmt(val, fmt_str, suffix=""):
    return (format(val, fmt_str) + suffix) if val is not None else "N/A"


def main():
    # ── Load results ────────────────────────────────────────────────────────
    rows = []
    baseline_tpot = None

    for display_name, method_key in METHODS:
        data = load_results(method_key)
        if data is None:
            print(f"  [skip] No results file for '{method_key}', run that experiment first.")
            continue
        rows.append((display_name, data))
        if method_key == "baseline":
            baseline_tpot = data.get("mean_decode_tpot_ms") or data.get("mean_tpot_ms")

    if not rows:
        print("No results found. Run at least one experiment script first.")
        return

    # ── Print table 1: user-visible latency and quality ────────────────────
    col_w = [32, 12, 14, 10, 10, 10]   # column widths
    header = (
        f"{'Method':<{col_w[0]}}"
        f"{'TTFT (ms)':>{col_w[1]}}"
        f"{'Decode TPOT':>{col_w[2]}}"
        f"{'Speedup':>{col_w[3]}}"
        f"{'Tok/sec':>{col_w[4]}}"
        f"{'Accuracy':>{col_w[5]}}"
    )
    divider = "-" * sum(col_w)

    if baseline_tpot is None:
        print("\n[note] baseline results not found, speedup column will show N/A.")
        print("       Run python src/run_client.py --method baseline first for speedup numbers.\n")

    print("Table 1: Latency and accuracy")
    print(divider)
    print(header)
    print(divider)

    for display_name, data in rows:
        ms_tok = data.get("mean_decode_tpot_ms") or data.get("mean_tpot_ms")
        ttft = data.get("mean_ttft_ms")
        tok_sec = data.get("mean_tokens_per_sec")
        # Show N/A rather than silently using 1.0x when baseline is missing.
        speedup = (baseline_tpot / ms_tok) if baseline_tpot and ms_tok else None
        acc_pct = data["accuracy"] * 100

        print(
            f"{display_name:<{col_w[0]}}"
            f"{_fmt(ttft, '.1f'):>{col_w[1]}}"
            f"{_fmt(ms_tok, '.1f'):>{col_w[2]}}"
            f"{_fmt(speedup, '.2f', 'x'):>{col_w[3]}}"
            f"{_fmt(tok_sec, '.1f'):>{col_w[4]}}"
            f"{_fmt(acc_pct, '.1f', '%'):>{col_w[5]}}"
        )

    print(divider)

    # ── Print table 2: tail latency, memory, and speculation details ───────
    col_w = [32, 12, 14, 14, 12, 14]
    header = (
        f"{'Method':<{col_w[0]}}"
        f"{'p95 TPOT':>{col_w[1]}}"
        f"{'Median TPOT':>{col_w[2]}}"
        f"{'GPU Mem (GB)':>{col_w[3]}}"
        f"{'Acceptance':>{col_w[4]}}"
        f"{'Total Tokens':>{col_w[5]}}"
    )
    divider = "-" * sum(col_w)

    print()
    print("Table 2: Tail latency and resource use")
    print(divider)
    print(header)
    print(divider)

    for display_name, data in rows:
        acceptance = data.get("mean_spec_acceptance_rate")
        acceptance_pct = acceptance * 100 if acceptance is not None else None
        print(
            f"{display_name:<{col_w[0]}}"
            f"{_fmt(data.get('p95_decode_tpot_ms'), '.1f'):>{col_w[1]}}"
            f"{_fmt(data.get('median_decode_tpot_ms'), '.1f'):>{col_w[2]}}"
            f"{_fmt(data.get('peak_gpu_memory_gb'), '.1f'):>{col_w[3]}}"
            f"{_fmt(acceptance_pct, '.1f', '%'):>{col_w[4]}}"
            f"{_fmt(data.get('total_tokens'), 'd'):>{col_w[5]}}"
        )

    print(divider)

    # ── Interpretation ───────────────────────────────────────────────────────
    print()
    print("Interpretation:")
    print("  A speedup > 1.3x at low concurrency is typical for well-matched draft models.")
    print("  Accuracy should stay within ±2 pp of baseline, larger gaps suggest a bug.")
    print("  Higher TPOT = slower decode; speculative decoding targets this number only.")


if __name__ == "__main__":
    main()
