# What Is Speculative Decoding?

A plain-English guide for anyone new to the topic. No prior knowledge of LLM internals required.

---

## The Slow Part of LLM Inference

When you ask a language model to generate text, it does not write the whole answer at once.
It writes **one token at a time**, left to right.

A token is roughly one word or one punctuation mark. The sentence "The cat sat on the mat"
is about 7 tokens.

To generate each token, the model runs a full **forward pass** through its neural network.
That means loading tens of billions of numbers (model weights), multiplying them together,
and producing a probability distribution over the vocabulary. Then it picks the most likely
next token, and repeats from the beginning.

This is called **autoregressive decoding**.

```
[Prompt: "The cat"]
        |
   Forward Pass #1     → outputs: "sat"
        |
   Forward Pass #2     → outputs: "on"
        |
   Forward Pass #3     → outputs: "the"
        |
   Forward Pass #4     → outputs: "mat"
        |
   Forward Pass #5     → outputs: <end>
```

Each arrow is a full forward pass through the entire model. Sequential, one after another.

The key bottleneck here is not raw compute, it is **memory bandwidth**. The GPU must load
the entire model's weights from memory for every single forward pass. For a 30B parameter
model in BF16, that is roughly 60 GB of data that must flow from memory to compute cores
for each token generated.

On the DGX Spark with 273 GB/s bandwidth, loading 60 GB takes about 220 ms per forward
pass in the worst case. In practice, other optimizations bring this down, but the
fundamental constraint is the same: memory bandwidth, not FLOPS, is the bottleneck.

---

## The Wasted GPU Capacity

Modern GPUs are designed for **massive parallelism**. The NVIDIA GB10 Blackwell on the
DGX Spark has thousands of compute cores that can execute operations simultaneously.

But autoregressive decoding uses this poorly.

When we generate token #1, we cannot start generating token #2 yet, because token #2
depends on what token #1 turned out to be. So the GPU crunches through a full forward pass,
produces one token, and then has to do it all over again.

Here is the critical insight that makes speculative decoding possible:

> **Verifying N tokens costs roughly the same compute as generating 1 token.**

Why? Because the forward pass is inherently parallel. Given a sequence of tokens, the
model can compute the probability of each position in parallel (this is how transformers
work, self-attention sees all positions at once).

So if someone hands the target model a sequence of N proposed tokens, it can check all N
of them in a single forward pass, the same cost as generating one token on its own.

That asymmetry is the entire foundation of speculative decoding.

---

## The Core Idea: Draft, Then Verify

Speculative decoding introduces a two-step process:

1. A **cheap, fast drafter** proposes several tokens ahead.
2. The **expensive target model** verifies all of them in one parallel forward pass.

The drafter can be much smaller and faster than the target, it just needs to be good
enough that the target agrees with it a reasonable fraction of the time.

Think of it like a junior analyst and a senior partner at a consulting firm.

The junior analyst (drafter) is fast, they draft a slide with four data points and an
interpretation. The senior partner (target model) reviews the whole slide at once and
either approves each point or corrects it.

Reviewing four points at once takes the senior partner about the same time as reviewing
one point from scratch. So if the junior analyst is even 60% right, the team ships more
work per unit of the senior partner's time.

In LLM terms: if the drafter proposes 4 tokens and the target accepts 3 of them, we just
got 3 tokens for the price of 1 forward pass.

---

## The Draft-Verify Loop

Here is what one step of speculative decoding looks like in practice:

```
Step 1: Drafter proposes
=============================================
Context: "The quick brown fox"

Drafter runs 4 small forward passes (fast, cheap):
  Proposed token 1: "jumps"
  Proposed token 2: "over"
  Proposed token 3: "the"
  Proposed token 4: "fence"    ← drafter guesses "fence"

=============================================

Step 2: Target verifies (ONE forward pass)
=============================================
Target model sees: "The quick brown fox [jumps] [over] [the] [fence]"
                                          ^       ^      ^     ^
                          Checks all 4 positions simultaneously

Target's own probabilities at each position:
  Position 1: "jumps", target also picks "jumps"  ✓ ACCEPT
  Position 2: "over" , target also picks "over"   ✓ ACCEPT
  Position 3: "the"  , target also picks "the"    ✓ ACCEPT
  Position 4: "fence", target picks "lazy"        ✗ REJECT

=============================================

Step 3: Combine accepted + corrected token
=============================================
Accepted: "jumps", "over", "the"
Rejected: "fence" → replaced by target's answer: "lazy"

Final output this step: "jumps over the lazy"
                         ^^^^^^^^^^^^^^^^
                         4 tokens, 1 target forward pass

=============================================

Step 4: Continue from "lazy"
=============================================
Next draft step starts from "The quick brown fox jumps over the lazy"
```

In a baseline system, this step would have produced exactly 1 token ("jumps") using 1
forward pass of the target. With speculative decoding, we produced 4 tokens using 1
forward pass. Even though we needed 4 small drafter passes, those are cheap enough that
the net result is a speedup.

**Important**: the fourth token is always the target model's own output. We never keep a
rejected draft token. The target always has the final word.

---

## Acceptance Rate: The Health Metric of Speculation

The **acceptance rate** is the fraction of draft tokens that the target model agrees with.

If the acceptance rate is 0.75, then on average the target accepts 3 out of every 4 draft
tokens.

Why does this matter for speedup? Here is the intuition:

```
num_speculative_tokens = 4

Acceptance rate 0.0  →  0 tokens accepted per step  →  net = 1 token (the correction)
Acceptance rate 0.5  →  2 tokens accepted per step  →  net = 3 tokens per step
Acceptance rate 0.75 →  3 tokens accepted per step  →  net = 4 tokens per step
Acceptance rate 1.0  →  4 tokens accepted per step  →  net = 5 tokens per step
```

When all 4 drafts are accepted, we also get the token after them (the target's output at
position 5) for free, so a perfect acceptance rate gives 5 tokens per step, not 4.

The metric you will see in the results is **mean accept length**, the average number of
tokens accepted per step. Our Eagle3 benchmarks show a mean accept length of ~2.32, which
translates to roughly 1.44× throughput gain over baseline.

A related metric is **TPOT (Time Per Output Token)**, the average wall-clock time to
produce one output token, measured in milliseconds. Lower TPOT = faster generation.

---

## Why Output Quality Is Unchanged

This is a common concern: does speculative decoding change the model's answers?

The answer is **no**, and it is mathematically provable.

Here is the intuition: whenever the target model rejects a draft token, it replaces it
with the token it would have generated itself. The rejected draft is discarded completely.

The acceptance criterion is not just "does the target agree?", it uses a statistical
technique (modified rejection sampling) that guarantees the accepted tokens have exactly
the same probability distribution as if the target had generated them alone.

In other words: if you ran the same prompt twice, once with speculation and once without
— the distribution of possible outputs is identical. The model's "personality" and
knowledge are unchanged. Only the speed of generation is different.

This is verified empirically in our benchmarks: we compare accuracy scores between
baseline and speculative methods. They should be within rounding error of each other.

---

## When Speculative Decoding Helps Most

Not every workload benefits equally. Here are the conditions where speculation shines:

**Single request (low batch size)**

Speculative decoding is most effective when the GPU is underutilized, one or two requests
running at a time. In this regime, the GPU has spare capacity that the drafter can use
without competing with other work.

**Long outputs**

The longer the output, the more draft-verify cycles happen, and the more the speedup
compounds. A 2000-token response benefits far more than a 20-token response.

**Repetitive or structured text**

If the output contains patterns, fixed phrasing, numbered lists, structured data, repeated
answer formats, the drafter is more likely to predict correctly. Higher acceptance rate =
more free tokens.

**Good draft-target alignment**

The drafter and target must use the same tokenizer and have similar output distributions.
A drafter from a completely different model family will have poor alignment and low
acceptance rates. That is why we use Qwen3-8B (same family as the Qwen3-32B target) for the
draft-target method, and Eagle3 (trained specifically on the Qwen3-32B's hidden states) for the
best results.

---

## When Speculative Decoding Does NOT Help

**High concurrency (large batch size)**

When the GPU is already saturated, many requests running in parallel, adding speculation
adds overhead without benefit. The target model's forward passes are already full of
useful work. In production serving with hundreds of concurrent users, batching is the
right optimization, not speculation.

**Very short outputs**

If the model only generates 10 tokens, the overhead of setting up speculation (loading
the draft model, running extra forward passes) may cost more than the speculation saves.

**High temperature sampling**

With temperature > 1.0, the model's outputs become more random and unpredictable. The
drafter cannot reliably guess what the target will choose. Acceptance rates collapse and
speculation provides little benefit.

**Mismatched draft and target**

If the draft model comes from a different model family with a different tokenizer or
vocabulary, many proposed tokens will be immediately rejected. This wastes the draft
compute without gaining any speedup.

---

## A Quick Summary

```
Standard autoregressive decoding:
  1 forward pass → 1 token
  1 forward pass → 1 token
  1 forward pass → 1 token
  ...

Speculative decoding:
  4 cheap draft passes (fast)
  1 expensive target pass (verifies all 4 simultaneously)
  Result: 2-5 tokens per target pass, depending on acceptance rate

Key constraints:
  - Output distribution is mathematically unchanged
  - Only useful at low batch sizes
  - Acceptance rate is the single most important health metric
  - Draft and target must share the same tokenizer
```

In this repo we demonstrate four different ways to implement this idea, ranging from the
simplest (no extra model needed) to the most sophisticated (a prediction head trained on
the target model's own internals). Each step teaches something new about the tradeoffs.

Continue reading: [02_ngram_explained.md](./02_ngram_explained.md)
