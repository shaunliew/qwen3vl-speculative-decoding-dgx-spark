# Method 4: Eagle3 Speculative Decoding

> **Learning goal**: Understand how a tiny prediction head trained on the target model's
> internal activations can outperform a full 16 GB draft model while using far less memory.

---

## What Makes Eagle3 Different

In Method 3 (draft-target), the draft model is a **separate, complete neural network** —
the 8B Qwen3-8B model runs independently and has no knowledge of what the Qwen3-32B is
"thinking." It predicts from its own weights, which are similar to the target's but not
identical. That gap is why acceptance rates plateau.

Eagle3 takes a completely different approach. Instead of a separate model, it adds a
**tiny prediction head** that plugs directly into the target model's internal layers.
This head sees the target model's hidden states, the rich intermediate representations
computed during each forward pass, and uses them to predict the next token before the
full output layer runs.

The head is not a separate model running independently. It is an extension of the target
model, accessing information that is already being computed as part of normal inference.
This is why its acceptance rates are higher: it predicts from what the Qwen3-32B already
"knows" at that point in the computation.

---

## What Are Hidden States?

Every transformer layer in a language model takes a sequence of floating-point vectors
as input and produces a transformed sequence as output. These intermediate vectors are
called **hidden states** or **residual stream activations**.

Here is a simplified picture of what happens inside the Qwen3-32B when it processes a token:

```
Input token IDs
      |
      v
[Embedding layer]  ← converts token IDs into dense vectors (e.g. 4096 floats per token)
      |
      v
[Transformer layer 1]  ← attention + feed-forward, updates the vector
      |
      v
[Transformer layer 2]  ← further refinement
      |
     ...
      |
      v
[Transformer layer N]  ← final hidden state: a 4096-float vector rich with context
      |
      v
[Vocabulary projection (lm_head)]  ← maps the vector to logits over 150,000+ tokens
      |
      v
Output token ID
```

The **hidden state after the final transformer layer** is the most informative vector in
the entire computation. It encodes:

- What the model has understood about the image
- What has been said so far in the conversation
- What the model intends to say next

The vocabulary projection (lm_head) then converts this rich vector into a probability
distribution over all possible next tokens. Eagle3 inserts its prediction head before
this final projection step, reading the same hidden state.

---

## How Eagle3 Is Trained

The Eagle3 head (`AngelSlim/Qwen3-32B_eagle3`) was trained in a supervised
fashion using the target model's own hidden states as inputs:

1. Run a large text dataset through the frozen Qwen3-32B model
2. At each position, record the hidden state at the final transformer layer
3. Train a small neural network (the Eagle3 head) to predict the next token from that hidden state
4. Because the head trains on the exact hidden states of the exact model it will speculate for,
   it learns to predict the target's outputs with high accuracy

The head is trained once and then frozen. During inference, it rides along with the target model
and proposes tokens ahead of the full vocabulary projection.

---

## ASCII Diagram: Normal Decode vs Eagle3

```
NORMAL DECODE (one token per forward pass):

  [Context tokens] → [Transformer layers 1..N] → [hidden state h_t]
                                                          |
                                              [lm_head projection]
                                                          |
                                              [probability distribution]
                                                          |
                                              sample → output token T_t


EAGLE3 SPECULATION (one target pass, multiple proposed tokens):

  [Context tokens] → [Transformer layers 1..N] → [hidden state h_t]
                                                       |        |
                                           [Eagle3 head, tiny]  [lm_head]
                                                       |        |
                                           proposed T_t+1      token T_t (paid)
                                                       |
                         [Eagle3 runs again on predicted hidden state]
                                                       |
                                           proposed T_t+2
                                                       |
                                           proposed T_t+3
                                                       |
                                           proposed T_t+4
                                                       |
  [Target verifies T_t+1 ... T_t+4 in ONE forward pass]
       ✓ ✓ ✗       ← accepts T_t+1, T_t+2, rejects at T_t+3
       |   |
  free  free  (only paid for T_t+3 correction)
```

The Eagle3 head is tiny: a single lightweight transformer decoder layer (with GQA attention and a feed-forward block) that operates on the hidden state rather than running a full separate model. Each proposal takes microseconds compared to a full forward pass. The target then verifies all proposals in one parallel pass, accepting or rejecting from left to right.

---

## Memory Advantage

This is where Eagle3 clearly beats draft-target on hardware efficiency:

| Component | Draft-Target | Eagle3 |
|-----------|-------------|--------|
| Target model (Qwen3-32B) | ~60 GB | ~60 GB |
| Draft / head model | ~16 GB (8B model) | ~1–2 GB (head only) |
| KV cache + activations | ~10–15 GB | ~10–15 GB |
| **Total** | **~86–91 GB** | **~71–77 GB** |

The Eagle3 head is orders of magnitude smaller than the 8B draft model because it is not
a complete transformer, it is a small network trained to operate on top of the target's
existing computation. You are not loading a second model; you are extending the first one.

On the DGX Spark with 128 GB unified memory, both approaches fit. But Eagle3 leaves
roughly 15 GB more headroom, which matters for:

- Longer KV cache (larger `max_model_len`)
- Higher `gpu_memory_utilization`
- Running other processes alongside inference

On more memory-constrained hardware, the 15 GB difference between Eagle3 and draft-target
could be the difference between fitting and not fitting.

---

## Claimed Numbers from AngelSlim

The AngelSlim team published benchmarks for the Eagle3 head paired with Qwen3-32B:

| Metric | Baseline | Eagle3 | Gain |
|--------|----------|--------|------|
| Throughput | 115 tok/s | 166 tok/s | **+44%** |
| Mean accept length | 1.0 (no speculation) | 2.32 tokens/step | 2.32× per target pass |

**What "mean accept length 2.32" means**: On average, each target verification pass accepts
2.32 draft tokens instead of just the 1 token baseline would produce. Every accepted draft
token is one you received without paying the cost of a target forward pass.

Compare to the geometric series formula from Method 3:

```
If mean accept length = 2.32, and we propose 4 tokens per step:

  E[tokens per step] ≈ 2.32 + 1 (the corrected token target always samples)
                     = 3.32 tokens per target forward pass

vs. baseline: 1 token per target forward pass
```

The 44% throughput improvement (1.44×) accounts for the Eagle3 head overhead. The raw
token gain per pass is larger, but the head itself takes a small amount of time for
each proposal.

---

## The Tradeoff: Trained Heads vs Flexibility

Eagle3's main limitation is the same as its main strength: **it must be trained per target model**.

The head in `AngelSlim/Qwen3-32B_eagle3` was trained specifically on
the hidden states of `Qwen/Qwen3-32B`. You cannot use this head with:

- A different quantized version of the same model (weights differ slightly)
- A fine-tuned version of the model (hidden states shift during fine-tuning)
- Any other model entirely

If you fine-tune the Qwen3-32B on your own dataset, the Eagle3 head trained on the base model
will have lower acceptance rates because the fine-tuned model's hidden states no longer match
what the head was trained on. In that case, draft-target with a compatible 8B model is your
only option until someone trains a new head on your fine-tuned model.

**Draft-target in contrast** works with any two models sharing a tokenizer, no training
required, no waiting for the community to release a head.

---

## Why Eagle3 Does Not Always Win on TPOT

Throughput (tokens per second across a batch) and TPOT (time per output token for a single
request) tell different stories.

Eagle3 is designed and benchmarked primarily for **throughput**, how many tokens can the
system generate per second when processing many requests in parallel. In a batch setting,
the Eagle3 head's low overhead and high acceptance rates shine.

For **single-request TPOT** (which is what you experience when chatting with a model
one question at a time), the picture is more nuanced:

- Eagle3's head adds latency to each target forward pass (small but real)
- The acceptance rate gain over draft-target may not fully compensate at small batch sizes
- Draft-target's 8B model, while heavier, generates proposals independently and can run
  slightly ahead of the target's verify pass in some implementations

On DGX Spark running one sample at a time (batch size = 1), you may see Eagle3 and
draft-target within a few percent of each other on TPOT, with Eagle3 pulling ahead on
throughput tests. Run `python src/run_client.py --method eagle3` and `python src/run_client.py --method draft_target`, then compare `results/eagle3.json` vs `results/draft_target.json` to see the
actual numbers on your hardware.

---

## vLLM Configuration

Inside the container (`nvcr.io/nvidia/vllm:26.04-py3`):

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-32B",
    gpu_memory_utilization=0.65,    # can go higher than draft-target, head is tiny
    max_model_len=8192,
    dtype="bfloat16",
    enable_prefix_caching=True,
    speculative_config={
        "method": "eagle",
        "model": "AngelSlim/Qwen3-32B_eagle3",  # the trained head
        "num_speculative_tokens": 4,
    },
)

params = SamplingParams(
    temperature=0,    # deterministic, required for accuracy comparison
    max_tokens=512,
    seed=42,
)
```

**Container requirement**: Eagle3 for VLMs (vision-language models) requires vLLM >= 0.12.0.
Use `nvcr.io/nvidia/vllm:26.04-py3`, which ships vLLM ~0.13.x. The earlier 26.01 container
ships a version that may not support Eagle3 with MoE VLMs.

You can verify the vLLM version inside the container:

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## Where to Find Eagle3 Heads

The AngelSlim team maintains a growing collection of trained Eagle3 heads on Hugging Face:

**Collection**: https://huggingface.co/collections/AngelSlim/eagle3

As of May 2026, the collection includes heads for several Qwen3 model sizes. Check the
collection page for the latest additions, new heads are released as new base models come out,
typically with a lag of a few weeks to months.

If a head does not exist for your model, your options are:

1. Use draft-target (Method 3) with a compatible smaller model
2. Train your own Eagle3 head using the Eagle3 training framework
3. Wait for the community to release one

For the purposes of this repo, we use the publicly available head for Qwen3-32B.

---

## Comparison Summary: Eagle3 vs Draft-Target

| Property | Draft-Target (Method 3) | Eagle3 (Method 4) |
|----------|------------------------|-------------------|
| Draft memory | ~16 GB (full 8B model) | ~1–2 GB (head only) |
| Acceptance rate | Moderate (depends on 8B quality) | High (trained on target's internals) |
| Flexibility | Any model pair sharing a tokenizer | Requires a pre-trained head |
| Works on fine-tuned models | Yes | Only if head was trained on fine-tune |
| Setup complexity | Low | Low (head downloads like any model) |
| Throughput gain | ~1.2–1.5× (varies) | ~1.44× (AngelSlim benchmark) |
| TPOT gain (single request) | Comparable | Comparable |

---

## Key Takeaways

1. Eagle3 adds a tiny prediction head that reads the target model's hidden states —
   the rich intermediate vectors computed before the final output projection.
2. Because it trains on the exact model it will speculate for, it predicts more accurately
   than an independent 8B draft model.
3. The Eagle3 head is ~1–2 GB, vs ~16 GB for the 8B draft model. It fits comfortably
   alongside the Qwen3-32B in DGX Spark's 128 GB unified memory.
4. AngelSlim's benchmarks show 2.32 mean accept length and 1.44× throughput for the
   Qwen3-32B head.
5. Eagle3 heads must be trained per target model. Draft-target is more flexible and
   works on day one, even for models with no published Eagle3 head.
6. Use `nvcr.io/nvidia/vllm:26.04-py3` (vLLM >= 0.12.0), earlier containers do not
   support Eagle3 for MoE VLMs.

---

## Putting It All Together

You have now seen all four methods:

| Method | Draft Source | Memory Cost | Acceptance Rate | When to Use |
|--------|-------------|-------------|-----------------|-------------|
| Baseline | None | ~60 GB |, | Reference only |
| N-gram | Prompt repetition | ~60 GB | Low-moderate | Structured, repetitive outputs |
| Draft-Target | Full 8B model | ~77 GB | Moderate | No Eagle3 head available |
| Eagle3 | Tiny trained head | ~62 GB | High | Head exists for your target model |

Run all four on the same 20 benchmark questions and compare `results/*.json` with
`src/benchmark_compare.py` to see which method gives you the best TPOT and acceptance
rate on your hardware.
