# Speculative Decoding Benchmark, Qwen3-32B on DGX Spark

Benchmarks four speculative decoding methods on a single NVIDIA DGX Spark using Qwen3-32B as the target model, Qwen3-8B as the draft model, and the AngelSlim Eagle3 head.

[Read the full write-up →](blog_draft.md)

---

## What This Repo Does

Runs the same 20 synthetic benchmark questions through four inference configurations and records TTFT, decode TPOT, speedup, and acceptance rate for each. Results are committed in `results/`. The four methods are: no speculation (baseline), n-gram pattern matching, draft-target with Qwen3-8B, and Eagle3 with the AngelSlim head. All four methods produce identical outputs at `temperature=0`, speculative decoding is lossless.

---

## Prerequisites

- NVIDIA DGX Spark (GB10 Blackwell, 128 GB unified memory)
- Docker
- ~84 GB free in `~/.cache/huggingface` for model weights
- `uv` installed on the host ([install guide](https://docs.astral.sh/uv/getting-started/installation/))

---

## Models Used

| Model | Role | Size |
|-------|------|------|
| Qwen/Qwen3-32B | Target (dense) | ~61 GB BF16 |
| Qwen/Qwen3-8B | Draft (Method 3) | ~16 GB BF16 |
| AngelSlim/Qwen3-32B_eagle3 | Eagle3 head (Method 4) | ~2 GB |

---

## Step 1: Download Models (Host Machine)

```bash
uv sync
bash setup.sh
```

This downloads ~84 GB total to `~/.cache/huggingface`. Run on the host before launching the container.

---

## Step 2: Launch the Container (Terminal 1)

```bash
docker run --gpus all --ipc=host \
  --name vllm-benchmark \
  -v $(pwd):/workspace \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:26.04-py3 \
  bash /workspace/start_server.sh baseline
```

Wait for "Application startup complete." in the logs before proceeding to Step 3.

---

## Step 3: Run the Benchmark (Terminal 2)

Open a second shell into the running container and run the benchmark client:

```bash
docker exec -it vllm-benchmark bash
cd /workspace

python src/run_client.py --method baseline
```

To run all four methods: stop the server with Ctrl+C in Terminal 1, restart it with the next method name, then re-run the client in Terminal 2. Repeat for each method.

```bash
# Shorthand for running all four in sequence (coordinating both terminals manually)
for method in baseline ngram draft_target eagle3; do
  # Terminal 1: bash /workspace/start_server.sh $method
  # Terminal 2:
  python src/run_client.py --method $method
done

# Print the comparison table once all four are done
python src/benchmark_compare.py
```

---

## Results (Qwen3-32B Dense)

All four methods produce 100% matching outputs at `temperature=0`, speculative decoding does not change answers, only speed.

| Method | TTFT (ms) | Decode TPOT (ms/tok) | Speedup | Tok/sec | Acceptance |
|--------|-----------|----------------------|---------|---------|------------|
| Baseline | 596 | 276.5 | 1.00x | 3.6 | N/A |
| N-gram | 345 | 243.0 | 1.14x | 4.2 | ~10–30% |
| Draft-target (Qwen3-8B) | 728 | 157.7 | 1.75x | 6.4 | 45–98.7% |
| Eagle3 (AngelSlim head) | 663 | 156.5 | 1.77x | 6.5 | ~13–25% |

---

## Benchmark Dataset

The benchmark uses 20 synthetic multiple-choice questions across math, science, history, and reasoning. See `data/questions.json` for the full list. Questions are designed to produce 200–400 token explanations, long enough to give speculative decoding meaningful rounds to amortize over.

---

## Repo Structure

```
src/
  question_loader.py    # 20 synthetic benchmark questions
  benchmark_harness.py  # data models and timing utilities
  run_client.py         # HTTP benchmark client (streams SSE for TTFT)
  benchmark_compare.py  # prints comparison table from results/
start_server.sh         # launches vllm serve for each method
setup.sh                # downloads model weights (run on host)
data/
  questions.json        # benchmark questions as JSON
results/
  baseline.json         # timing results for each method
  ngram.json
  draft_target.json
  eagle3.json
concepts/               # educational markdown explanations
notebooks/
  walkthrough.ipynb     # interactive walkthrough
```

---

## Acceptance Rate

Acceptance rate is visible in the Terminal 1 server logs. Search for `SpecDecoding metrics`. Higher acceptance means more draft tokens were accepted per forward pass, which produces more speedup. N-gram does not report this metric. Draft-target and Eagle3 both do.

---

## Learn More

- `concepts/01_what_is_speculative_decoding.md`
- `concepts/02_ngram_explained.md`
- `concepts/03_draft_target_explained.md`
- `concepts/04_eagle3_explained.md`
- `notebooks/walkthrough.ipynb`
- [blog_draft.md](blog_draft.md), full analysis including MoE vs dense comparison

---

## References

- [vLLM Speculative Decoding Docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- [AngelSlim Eagle3 Collection](https://huggingface.co/collections/AngelSlim/eagle3)
- [Qwen3-32B Model Card](https://huggingface.co/Qwen/Qwen3-32B)
- [NVIDIA DGX Spark vLLM Guide](https://build.nvidia.com/spark/speculative-decoding)
- [BentoML Speculative Decoding Blog](https://www.bentoml.com/blog/3x-faster-llm-inference-with-speculative-decoding)
