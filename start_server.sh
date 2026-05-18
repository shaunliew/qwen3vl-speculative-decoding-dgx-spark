#!/usr/bin/env bash
# start_server.sh: Launch vllm serve for a specific speculative decoding method.
#
# Usage:
#   bash start_server.sh baseline
#   bash start_server.sh ngram
#   bash start_server.sh draft_target
#   bash start_server.sh eagle3
#
# Run this in Terminal 1 (inside the container). Keep it running.
# In Terminal 2, run: python src/run_client.py --method <same-method>
#
# The server is ready when you see:
#   "Application startup complete."
#
# To watch acceptance rate in real time, search the server logs for "spec_decode":
#   e.g.  "spec_decode_acceptance_rate: 0.87"
# This appears per-request when speculative decoding is active. Baseline shows nothing here.

set -e  # Exit immediately if any command fails, prevents silent half-starts

# ── Argument validation ──────────────────────────────────────────────────────
if [ -z "$1" ]; then
    echo "Usage: bash start_server.sh <method>"
    echo "  method = baseline | ngram | draft_target | eagle3"
    exit 1
fi

METHOD="$1"

# ── Shared flags (used by every method) ──────────────────────────────────────
#
# --model: Dense model: all 32B params active per token. Unlike the MoE variant, 32B > 8B (draft)
#          so the size ratio is correct for speculative decoding.
#          Fits in DGX Spark's 128 GB unified memory with room for a draft model.
#
# --gpu-memory-utilization 0.75: Reserve 75% of GPU memory for models + KV cache.
#          Start conservative, increase to 0.75 if stable, 0.85 if no OOM.
#          DGX Spark has 128 GB unified, so 60% still gives ~76 GB for the model + KV.
#
# --max-model-len 8192: Cap the KV cache at 8192 tokens.
#          Qwen3 supports 128K natively, but benchmark prompts are short.
#          Smaller context = smaller KV cache = more headroom for draft model.
#
# --dtype bfloat16: GB10 (Blackwell) has native BF16 hardware support.
#          BF16 is ~half the memory of FP32 with no meaningful accuracy loss for inference.
#
# --allowed-local-media-path: not needed for text-only models (no image loading).
#
# --port 8000: Standard OpenAI-compatible REST API port.
#          The client (run_client.py) connects to http://localhost:8000 by default.
#
# Request logging: ON by default, no flag needed.
#          Per-request log lines show acceptance stats for speculative methods.
#          Search "spec_decode_acceptance_rate" in Terminal 1 while the client runs.

BASE_FLAGS=(
    "Qwen/Qwen3-32B"
    # 0.75 = 96 GB budget on DGX Spark's 128 GB unified memory.
    # Baseline and n-gram use ~57-63 GB (models only) leaving ~33-39 GB for KV cache.
    # Draft-target loads two models (~72 GB total), leaving ~24 GB for KV cache.
    # At 0.6 (76.8 GB budget), draft-target OOMs: only 4.66 GB left after both models.
    --gpu-memory-utilization 0.75
    --max-model-len 8192
    --dtype bfloat16
    --port 8000
    # Disable Qwen3's thinking mode globally for all requests.
    # Qwen3 enables extended reasoning (<think> blocks) by default. Without this flag,
    # the model generates a long internal reasoning chain before answering, consuming
    # all available tokens and never reaching the actual answer letter.
    # This server-side flag is simpler than passing enable_thinking=False per-request
    # in every API call. Per-request temperature=0 still overrides generation_config.json
    # defaults (per-request params always win over server defaults in vLLM).
    --default-chat-template-kwargs '{"enable_thinking": false}'
    # Request logging is ON by default in vLLM, no flag needed.
    # Per-request logs show acceptance stats: search "spec_decode" in Terminal 1.
)

# ── Method-specific speculative decoding configuration ───────────────────────
case "$METHOD" in

  baseline)
    # No speculative config, pure autoregressive decoding.
    # Every token is generated sequentially: sample → wait → sample → wait ...
    # The GPU sits idle between tokens while we wait for the previous one.
    # This is the reference baseline we compare everything else against.
    echo "Starting vllm serve, METHOD: baseline (no speculation), Qwen3-32B (dense)"
    echo "Expect: nothing in logs related to spec_decode, this is the control."
    vllm serve "${BASE_FLAGS[@]}"
    ;;

  ngram)
    # N-gram speculation: no extra model needed.
    # vLLM scans the existing context for repeated n-gram patterns.
    # If the context contains "the cat sat on" and the model just generated "the cat",
    # vLLM proposes "sat on" as draft tokens, completely free, no GPU compute.
    #
    # num_speculative_tokens 4: propose up to 4 tokens per draft step.
    # prompt_lookup_min/max: minimum and maximum n-gram length to search for.
    #   Shorter min (2) catches more repetitions; longer max (5) reduces false matches.
    #
    # Best case for n-gram: structured outputs like "Answer: A", "Answer: B"
    # repeating across many questions, the pattern appears in the context window.
    echo "Starting vllm serve, METHOD: ngram speculation"
    echo "Watch for: spec_decode_acceptance_rate in server logs. Higher = more free tokens."
    vllm serve "${BASE_FLAGS[@]}" \
        --speculative-config '{
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 5
        }'
    ;;

  draft_target)
    # Draft-target speculation: a separate smaller VLM (8B) proposes tokens.
    # The key insight: verifying N tokens costs roughly the same compute as generating 1.
    # So if the 8B draft proposes 4 tokens and the 30B accepts 3, we got 3 tokens
    # for the compute cost of 1 baseline token, that's the "free speedup."
    #
    # Memory: ~61 GB (32B dense) + ~16 GB (8B) = ~77 GB total.
    # This fits in DGX Spark's 128 GB but would be impossible on a 24 GB RTX 4090.
    #
    # Why Qwen3-8B works as draft: same tokenizer family as the 32B target.
    # Draft tokens are always valid vocabulary entries for the 32B to verify.
    # Unlike Qwen3-30B-A3B MoE where draft was heavier than target, 8B is genuinely
    # lighter than 32B dense, so the size ratio is correct for speculative decoding.
    # NOTE: draft_model is fully supported for TEXT-ONLY models.
    # It was blocked only for VLMs (Qwen3-VL) in the NVIDIA container, the reason
    # we switched to text-only Qwen3. Qwen3-8B as draft for Qwen3-32B works correctly.
    echo "Starting vllm serve, METHOD: draft_target (Qwen3-8B draft → Qwen3-32B verify)"
    echo "Both models load simultaneously, expect ~77 GB total (32B dense + 8B draft)."
    echo "Watch for: spec_decode_acceptance_rate. Target >0.7 for meaningful speedup."
    vllm serve "${BASE_FLAGS[@]}" \
        --speculative-config '{
            "method": "draft_model",
            "model": "Qwen/Qwen3-8B",
            "num_speculative_tokens": 5
        }'
    ;;

  eagle3)
    # Eagle3 speculation: a lightweight head trained on the 30B A3B's internal hidden states.
    # Unlike draft-target (which uses an independent 8B model), Eagle3 sees what the
    # target model "is thinking", its intermediate layer activations, and predicts
    # the next token from that richer signal. This is why it achieves higher acceptance.
    #
    # This is the dense version, trained on Qwen3-32B (not the VL or MoE variant).
    # Throughput: 1.66-1.85x on H20 (AngelSlim benchmarks).
    #
    # Memory: ~66 GB (32B dense) + ~1-2 GB (Eagle3 head) = ~68 GB total.
    # Much more memory-efficient than draft-target's full 8B.
    #
    # Requires vLLM >= 0.12.0, use container nvcr.io/nvidia/vllm:26.04-py3 or later.
    echo "Starting vllm serve, METHOD: eagle3 (AngelSlim trained speculation head, dense 32B)"
    echo "Smallest memory footprint of all speculative methods (~68 GB total)."
    echo "Watch for: spec_decode_acceptance_rate, expect >0.8 for Eagle3."
    vllm serve "${BASE_FLAGS[@]}" \
        --speculative-config '{
            "method": "eagle3",
            "model": "AngelSlim/Qwen3-32B_eagle3",
            "num_speculative_tokens": 5
        }'
    ;;

  *)
    echo "Error: unknown method '$METHOD'"
    echo "Usage: bash start_server.sh <method>"
    echo "  method = baseline | ngram | draft_target | eagle3"
    exit 1
    ;;
esac

# ── Reading the server logs ───────────────────────────────────────────────────
# The server prints one line per completed request when --disable-log-requests false.
# For speculative methods, look for lines containing "spec_decode", e.g.:
#
#   INFO ... spec_decode_acceptance_rate=0.87 spec_decode_draft_acceptance_rate=0.91
#
# acceptance_rate: fraction of all draft tokens accepted (0.0–1.0)
# draft_acceptance_rate: token-level acceptance before any correction step
#
# The client (run_client.py) captures per-request acceptance rates where vLLM
# exposes them via the streaming API. Server logs are the fallback source of truth.
