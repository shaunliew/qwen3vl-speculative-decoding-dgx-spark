# Method 3: Draft-Target Speculative Decoding

> **Learning goal**: Understand how running a small "helper" model alongside the large model
> can dramatically reduce the time you wait for each token, without changing the output.

---

## The Core Idea

Normal inference is sequential. The 32B dense model generates one token, sends it back to
you, generates another, and so on. Every token requires a full forward pass through the
giant model. That is expensive.

Draft-target speculation breaks that chain. Instead of asking the big model to generate
every token, you use a **small, fast draft model** to propose a batch of candidate tokens,
then ask the big model to **verify** the whole batch in one shot.

In our setup:

- **Draft model**: `Qwen/Qwen3-8B`, 8 billion parameters, fast
- **Target model**: `Qwen/Qwen3-32B`, 32 billion parameters, slower but authoritative
- **Draft budget**: 4 tokens proposed per step (`num_speculative_tokens = 4`)

The draft model runs 4 fast sequential forward passes and proposes tokens T1, T2, T3, T4.
The target model then runs **one** forward pass to verify all four, accepting or rejecting
each one from left to right. If three are accepted, you received 3 tokens for roughly the
cost of 1 target-model forward pass. That is the speedup.

---

## Why Verification Is Cheap

This is the key insight behind all speculative decoding. Burn this into your memory:

> **The target model's forward pass costs the same whether it generates 1 token or
> checks 4 draft tokens and generates 1 corrected token.**

Here is why. During a standard forward pass, the transformer processes all input tokens
in **parallel**. The attention mechanism, feed-forward layers, and final projection all
operate on a sequence at once. When you verify 4 draft tokens, you simply append them to
the input sequence and run one forward pass, the model computes logit distributions at
every position simultaneously, and you compare those distributions to what the draft
proposed.

If the draft said T1 and the target agrees (T1 has high probability at that position),
you accept T1 for free. If it disagrees at T3, you reject T3, sample the correct token
from the target's distribution, and discard T4 (since T4 was conditioned on the wrong T3).

The extra cost over baseline is:
1. Running the draft model to generate 4 proposals (real but small cost)
2. Slightly longer input to the target's verification pass (negligible)

Everything else is free acceleration.

---

## ASCII Diagram: One Speculation Step

```
STEP 1, Draft model proposes 4 tokens (sequential mini-passes):

  Context so far: [img_tokens... "The answer is"]
                                              |
  8B Draft:  pass 1 → proposes  T1 = "B"
  8B Draft:  pass 2 → proposes  T2 = ","
  8B Draft:  pass 3 → proposes  T3 = "because"
  8B Draft:  pass 4 → proposes  T4 = "the"

  Draft output: ["B", ",", "because", "the"]


STEP 2, Target model verifies all 4 in ONE forward pass:

  Input to target: [...context... | "B" | "," | "because" | "the"]
                                    ↑      ↑       ↑            ↑
  Target checks:                  ✓ ok  ✓ ok    ✗ reject     (skip)
                                    T1     T2      T3           T4

  At T3, target disagrees: samples correct token T3' = "of"


STEP 3, Net result:

  Tokens accepted this step:  "B", ","        (2 free tokens)
  Token from target sample:   "of"            (1 paid token)
  Total output: ["B", ",", "of"]              (3 tokens for ~1 target forward pass cost)

  Baseline would have produced: ["B"]         (1 token for 1 target forward pass cost)

  Effective speedup this step: 3×
```

---

## Why the Same Model Family Matters

One rule for draft-target: **the draft and target models must share the same tokenizer**.

A tokenizer converts raw text into token IDs, integer indices into a vocabulary table.
If the 8B draft uses vocabulary ID 1234 for the word "because" but the 32B target maps "because"
to vocabulary ID 5678, then every draft token is meaningless garbage to the target. The
verification step would reject everything, and you would have zero speedup plus wasted compute.

Because both our models are from the Qwen3 family:

- Same tokenizer: `Qwen/Qwen3-32B` and `Qwen/Qwen3-8B` use identical vocabulary tables and tokenization rules
- Same special tokens: `<|im_start|>` and `<|im_end|>` have the same IDs in both models

This is not just a convenient property, it is a hard requirement. Mixing models from
different families (e.g., a Llama draft with a Qwen target) would silently produce
nonsense outputs because every draft token ID maps to a different word in the target's
vocabulary.

---

## Acceptance Rate and Speedup Math

Let alpha (α) be the **acceptance rate**, the probability that any single draft token
is accepted by the target model. The expected number of tokens produced per target
forward pass follows a geometric series:

```
E[tokens per step] = 1 + α + α² + α³ + α⁴

(The "+1" at the end is the corrected token the target always samples.)
```

For `num_speculative_tokens = 4` and a realistic acceptance rate of α = 0.7:

```
E[tokens] = 1 + 0.7 + 0.49 + 0.343 + 0.2401
           = 2.77 tokens per target forward pass

vs. baseline = 1.0 token per target forward pass

Raw token gain: 2.77×
```

For α = 0.8 (a good draft model):

```
E[tokens] = 1 + 0.8 + 0.64 + 0.512 + 0.4096
           = 3.36 tokens per target forward pass
```

For α = 0.5 (a poor draft model):

```
E[tokens] = 1 + 0.5 + 0.25 + 0.125 + 0.0625
           = 1.94 tokens per target forward pass
```

**Key takeaway**: Acceptance rate is the single most important number in draft-target
speculation. Higher acceptance rate = bigger speedup. If α drops below ~0.5, the
overhead of running the draft model may outweigh the gains.

---

## The Draft Model Overhead (Why Net Speedup Is Smaller)

The raw token gain above assumes the draft model is free. It is not.

The 8B draft runs 4 sequential passes before each target verification. The 8B model is
roughly 4x smaller than the Qwen3-32B target (8B vs 32B parameters), but it
is not 4× faster because memory bandwidth and attention costs do not scale perfectly with
parameter count on unified memory hardware.

A rough estimate for the draft overhead on DGX Spark:

```
Effective TPOT with speculation =
    (time for 4 draft passes + time for 1 target verify pass)
    / E[tokens accepted + 1]
```

If 4 draft passes take roughly the same wall-clock time as 1 baseline target pass, and
each target pass cost is T_target:

```
Effective TPOT ≈ (T_draft_total + T_target) / 2.77

If T_draft_total ≈ 0.5 × T_target:
  Effective TPOT ≈ (0.5 + 1.0) × T_target / 2.77
                 ≈ 0.54 × T_target
  → Speedup ≈ 1.8×
```

The real numbers on DGX Spark will differ, run `python src/run_client.py --method draft_target` to measure.
The benchmark harness captures actual TPOT, acceptance rate, and mean accept length.

---

## Memory on DGX Spark

Loading both models simultaneously is the other key requirement. Here is the memory math:

| Model | Parameters | Memory (BF16) |
|-------|-----------|---------------|
| Qwen3-32B | 32B (dense, all active) | ~61 GB |
| Qwen3-8B | 8B | ~16 GB |
| KV cache + activations | | ~10 GB |
| **Total** | | **~87 GB** |

DGX Spark has 128 GB of unified memory shared between CPU and GPU. This fits comfortably.

Compare to a typical discrete GPU:

```
RTX 4090:  24 GB VRAM
Qwen3-32B: ~61 GB  ← already 2.5x over the GPU limit
8B draft:  +16 GB
Total:     ~77 GB  ← impossible on a single GPU

DGX Spark: 128 GB unified
Total:     ~87 GB  ← fits with 41 GB to spare for OS, KV cache, and other processes
```

This is the hardware story of the project. The DGX Spark's unified memory architecture
is what makes loading two large models on a single device feasible. You cannot do this
on consumer or prosumer GPU hardware without model quantization or model offloading tricks.

---

## When Draft-Target Beats Eagle3

Eagle3 (Method 4) is generally faster than draft-target when a trained head is available.
But draft-target has one major advantage: **it is model-agnostic within a tokenizer family**.

Use draft-target when:

- No Eagle3 head has been trained for your target model
- You want to experiment with different draft models (e.g., compare 3B vs 7B vs 8B drafts)
- The target model is newer and the Eagle3 community has not yet released a trained head
- You want a drop-in solution that works with any two models that share a tokenizer

Eagle3 requires someone to have trained a prediction head specifically on your target model's
internals. As of May 2026, the AngelSlim team has released heads for several Qwen3 models,
but coverage will always lag new model releases by weeks or months.

Draft-target works on day one, with any compatible draft model you can find.

---

## vLLM Configuration

Inside the container (`nvcr.io/nvidia/vllm:26.04-py3`):

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-32B",
    gpu_memory_utilization=0.6,     # leaves headroom for the 8B draft
    max_model_len=8192,             # cap context to avoid KV cache OOM
    dtype="bfloat16",               # native BF16 on GB10
    enable_prefix_caching=True,     # helps when prompts share repeated structure
    speculative_config={
        "method": "draft_model",
        "model": "Qwen/Qwen3-8B",  # the draft model
        "num_speculative_tokens": 4,             # propose 4 tokens per step
    },
)

params = SamplingParams(
    temperature=0,    # deterministic, required for accuracy comparison
    max_tokens=512,
    seed=42,
)
```

The `gpu_memory_utilization=0.6` is lower than baseline (0.6 vs 0.7) because you need
to reserve memory for the 8B draft model in the same unified memory pool. vLLM manages
both models internally when `method: "draft_model"` is set.

---

## What to Look For in the Results

When you run `python src/run_client.py --method draft_target` and compare to `results/baseline.json`, watch for:

| Metric | What it tells you |
|--------|-------------------|
| Acceptance rate | How well the 8B draft predicts 30B A3B's outputs |
| Mean accept length | Average tokens accepted per verification pass |
| TPOT (ms/token) | Whether the speedup covers the draft overhead |
| Output accuracy | Should match baseline exactly, speculation never changes correctness |

If acceptance rate is below 0.5, the 8B and 30B A3B are diverging in their predictions —
possibly because the image content is unusual, the question requires rare knowledge, or
the temperature is non-zero (always use temperature=0 for these comparisons).

---

## Key Takeaways

1. The draft model proposes; the target model verifies. Verification is the same cost as
   baseline generation, everything accepted is a free speedup.
2. Tokenizers must match exactly. Qwen3-8B and Qwen3-32B share the same vocabulary
   and vision encoding, making them a valid pair.
3. Acceptance rate α drives speedup more than anything else. Measure it on your actual inputs.
4. The 8B draft adds real overhead, net speedup is smaller than raw token gain math suggests.
5. On DGX Spark, both models fit in 128 GB unified memory. On a 24 GB GPU, this is impossible.
6. Draft-target is model-agnostic within a tokenizer family, it works even when no Eagle3
   head has been trained for your target model.

Next: [04_eagle3_explained.md](./04_eagle3_explained.md), how Eagle3 replaces the full 8B
draft model with a tiny prediction head trained on the target's own hidden states.
