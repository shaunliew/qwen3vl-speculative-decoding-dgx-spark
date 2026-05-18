# N-gram Speculation: The Free Speedup

This is the simplest form of speculative decoding, no extra model, no extra GPU memory.
Just pattern matching on text the model has already seen.

---

## What Is an N-gram?

An **n-gram** is simply a sequence of N consecutive tokens.

- 1-gram (unigram): `["cat"]`
- 2-gram (bigram): `["the", "cat"]`
- 3-gram (trigram): `["the", "cat", "sat"]`
- 4-gram: `["the", "cat", "sat", "on"]`

That is it. Nothing fancy, just consecutive chunks of a token sequence. The "n" refers
to how many tokens are in the chunk.

N-gram models have been around in NLP for decades. We are not using them to generate text
here, we are using them to **look up predictions** from text that already exists in the
context window.

---

## How N-gram Speculation Works

When the target model is about to generate the next token, n-gram speculation first asks:

> "Has this exact sequence of tokens appeared earlier in the context? If so, what came after it?"

Here is a concrete example. Suppose the model is answering multiple-choice questions and has already
generated several answers:

```
Context so far:
  ... Question 1: [image] Which planet is largest?
  The answer is: Jupiter

  Question 2: [image] What is the capital of France?
  The answer is: Paris

  Question 3: [image] Which element has symbol Fe?
  The answer is: [← model is here, about to generate]
```

The model just generated the tokens `["The", "answer", "is", ":"]`. The n-gram lookup
scans the earlier context and finds that exact sequence appeared twice before:

- After `["The", "answer", "is", ":"]` → `"Jupiter"`
- After `["The", "answer", "is", ":"]` → `"Paris"`

The most recent match was `"Paris"`, so the drafter proposes `"Paris"` as the next token.
The target model then verifies this, if the actual answer is `"Iron"`, it rejects `"Paris"`
and uses `"Iron"` instead. If the answer happened to be `"Paris"` again, it accepts for free.

In practice, the repetitive structure of "The answer is: X" means the drafter will sometimes
guess wrong on the specific answer, but it will usually get the phrasing right. That is
valuable because the phrasing tokens ("The", "answer", "is", ":") are accepted for free.

---

## Why N-gram Is "Free"

With a draft-target method (Method 3), we load a full second model (8B parameters, ~16 GB).
That takes GPU memory and compute time on every step.

With n-gram speculation, the drafter is nothing more than a **string search** over the
current token buffer. There is no neural network involved. The cost is:

- No extra GPU memory beyond what the target already uses
- A few microseconds of CPU time for the pattern lookup
- Zero additional model weights to load

This makes n-gram speculation the lowest-cost way to get some speculative speedup.

---

## The Three Configuration Parameters

When you call vLLM with n-gram speculation, you set three numbers:

```python
speculative_config={
    "method": "ngram",
    "num_speculative_tokens": 4,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 5,
}
```

**`num_speculative_tokens` (default: 4)**

How many tokens the drafter proposes in each step. Setting this to 4 means the drafter
proposes 4 tokens, and the target verifies all 4 in a single forward pass.

Higher values mean more potential free tokens per step, but also more wasted work when
the sequence diverges from the context. On structured short-answer tasks, 4 is a
reasonable default.

**`prompt_lookup_min` (default: 2)**

The minimum n-gram length the drafter will search for. Setting this to 2 means the drafter
will match on sequences as short as 2 tokens.

A shorter minimum means more matches, even short repeated phrases will trigger speculation.
But short matches are less specific: the sequence `["is", ":"]` might match many places in
the context, giving less accurate predictions.

**`prompt_lookup_max` (default: 5)**

The maximum n-gram length to search for. Setting this to 5 means the drafter looks for
sequences up to 5 tokens long.

Longer n-grams are more specific, if you find a 5-token match, the prediction is likely
to be accurate because fewer things in the context have that exact 5-token prefix. The
drafter tries the longest match first and falls back to shorter ones.

```
Lookup process with prompt_lookup_min=2, prompt_lookup_max=5:
  Given current tail: ["The", "answer", "is", ":", ...]
  
  Try 5-gram: ["The", "answer", "is", ":", X]   → search for exact match in context
  If found → propose the token that followed it
  If not found → try 4-gram: ["answer", "is", ":", X]
  If not found → try 3-gram: ["is", ":", X]
  If not found → try 2-gram: [":", X]
  If not found → no n-gram draft this step (fall back to target)
```

---

## When N-gram Excels: Structured Outputs

N-gram speculation thrives when the model generates text that echoes patterns already in
the context. Structured multiple-choice benchmarks are a good example.

Our synthetic benchmark presents multiple-choice questions. A
typical answer might look like:

```
The answer is B. The image shows a mitochondria, which is the
powerhouse of the cell, not a nucleus (A), chloroplast (C),
or ribosome (D).
```

The phrasing "The answer is" appears in almost every response. After processing a dozen
questions, the context is full of this exact sequence. The n-gram drafter will propose
"The", "answer", "is" correctly almost every time.

Even if it guesses the wrong letter (proposing "B" when the answer is "C"), the drafter
still got 3 out of 4 tokens right, and those tokens are accepted for free.

Other cases where n-gram works well:

- Numbered lists: `"1.", "2.", "3."`, the structure repeats
- Fixed formats: `"Option A:", "Option B:"`, character labels repeat
- Boilerplate text: footers, disclaimers, repeated instructions
- Code with repeated patterns: `import`, `def`, `return` in similar positions

---

## When N-gram Fails: Novel Outputs

N-gram speculation cannot predict tokens it has never seen in the context. If the model
is generating a new scientific explanation, a creative story, or a long reasoning chain,
the specific tokens used are unlikely to repeat verbatim.

```
Prompt: "Explain how mitochondria produce ATP."

Response (first question, empty context history):
  "Mitochondria produce ATP through a process called oxidative
   phosphorylation, which occurs in the inner mitochondrial
   membrane..."
```

On the first question, there are no prior patterns to match. The n-gram drafter sits idle.
On the second question about the same topic, some phrases might repeat, but science
answers rarely repeat word-for-word.

This is not a failure of the technique, it is working as intended. When there are no
useful n-gram matches, the system simply falls back to standard target-only generation.
No tokens are wasted; we just get no speedup on those steps.

---

## Acceptance Rate on Our Benchmark: What to Expect

On structured multiple-choice questions, n-gram speculation typically achieves an acceptance
rate in the range of **20%–50%**, depending on how structured the answers are.

That range might sound modest. But let us work through the math to see why it still helps.

```
Setup:
  num_speculative_tokens = 4
  baseline TPOT = 8 ms/token
  target forward pass cost ≈ 8 ms (same as baseline)

Scenario: acceptance rate = 0.50

Per draft-verify step:
  Tokens proposed = 4
  Expected accepted = 4 × 0.50 = 2 tokens
  Plus the correction token = 1 token
  Net tokens per step = 3

Baseline tokens per step = 1

Speedup = 3 tokens / 1 token = 3×?

No, we also need to account for drafter overhead.
N-gram lookup is near-zero cost, so we can ignore it.
The real bottleneck is the target forward pass.

Effective TPOT with n-gram:
  = target_pass_time / net_tokens_per_step
  = 8 ms / 3 tokens
  = 2.7 ms per token

Wait, that implies a 3× speedup from 50% acceptance...
```

The math checks out because n-gram has virtually zero overhead. Unlike the draft-target
method (which pays for 4 small model forward passes per step), n-gram's "draft cost" is
negligible.

In practice, the observed speedup is typically **1.1x-1.4x** on structured question benchmarks. The gap between
theory and practice comes from:

- Not every step has an n-gram match (some steps fall back to baseline)
- KV cache overhead grows as context length increases
- The acceptance distribution is not perfectly uniform across all steps

Still, a 1.2×–1.5× throughput improvement with zero additional GPU memory is a compelling
tradeoff for this workload.

---

## How to Read the Results

When you run `python src/run_ngram.py` and then `python src/benchmark_compare.py`, you
will see output like:

```
Method        TPOT (ms)   Speedup   Accept Rate   Mean Accept Len
-----------   ---------   -------   -----------   ---------------
baseline         8.2 ms      1.0×          n/a               n/a
ngram            6.1 ms      1.3×        38.0%              1.91
```

Reading this table:

- **TPOT dropped from 8.2 ms to 6.1 ms**, each output token now takes 1.9 ms less to
  produce. Over a 200-token response, that is 380 ms saved. For a real user, this is the
  difference between a response appearing "fast" versus "snappy."

- **Speedup of 1.3×**, for every 1 second of baseline generation, n-gram finishes in
  ~0.77 seconds. Modest but real, and it cost nothing extra.

- **Acceptance rate 38%**, on average, 38% of proposed tokens were accepted by the target.
  The other 62% were rejected and replaced with the target's own token.

- **Mean accept length 1.91**, each draft-verify cycle produced an average of 1.91 tokens,
  instead of 1.0 token in the baseline. With 4 tokens proposed, we would need a 100%
  acceptance rate to reach 5.0, 1.91 reflects the 38% acceptance rate compounded over
  a 4-token proposal.

The formula connecting these:

```
mean_accept_length ≈ (1 - acceptance_rate^(num_speculative_tokens+1))
                     / (1 - acceptance_rate)

At acceptance_rate=0.38, num_speculative_tokens=4:
  = (1 - 0.38^5) / (1 - 0.38)
  = (1 - 0.0079) / 0.62
  ≈ 1.60   (theoretical, geometric series)

Observed 1.91 is slightly higher because the acceptance pattern
is not perfectly i.i.d., some positions have higher accept rates
than others (e.g. the structured "The answer is" phrasing).
```

---

## Summary

N-gram speculation is the entry point to understanding speculative decoding because it
strips away everything except the core mechanism: propose tokens from known patterns,
verify them in parallel, accept the good ones for free.

```
No extra model       → zero GPU memory overhead
String matching      → drafter cost is microseconds
Structured outputs   → good acceptance on multiple-choice style benchmarks
Novel outputs        → graceful fallback to baseline

Best suited for:
  Low concurrency (1-2 requests)
  Structured, repetitive answer formats
  Long outputs where patterns accumulate in context

Not suited for:
  High concurrency (GPU already saturated)
  Creative or unpredictable outputs
  Very short generations (<20 tokens)
```

Once you have understood n-gram, the jump to draft-target (Method 3) is conceptual, not
mechanical, you just replace the string-matching drafter with a small neural network.

Continue reading: [03_draft_target_explained.md](./03_draft_target_explained.md)
