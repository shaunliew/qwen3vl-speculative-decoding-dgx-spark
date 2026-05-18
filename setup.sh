#!/usr/bin/env bash
# Run this on the HOST machine BEFORE launching the Docker container.
# Downloads all model weights to the HuggingFace cache so the container
# can mount them without needing internet access inside Docker.
#
# Prerequisites:
#   uv sync               , creates .venv/ with huggingface-hub[cli]
#   uv run hf auth login  , authenticate if models are gated

set -e

# ── Preflight: check uv is available ────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install it from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Ensure the venv is created and huggingface-hub[cli] is installed
uv sync --quiet

echo "=== Qwen3 Speculative Decoding, Model Download ==="
echo "Destination: \$HF_HOME (default: ~/.cache/huggingface)"
echo ""

# ── Model 1: Primary target model ───────────────────────────────────────────
# Qwen3-32B  ~66 GB BF16 on disk
# Dense model: all 32B params active per token. Correct size ratio for speculative decoding with 8B draft.
echo "[1/3] Downloading Qwen/Qwen3-32B  (~66 GB)..."
uv run hf download Qwen/Qwen3-32B
echo "      Done."
echo ""

# ── Model 2: Draft model for Method 3 (draft-target speculation) ────────────
# Qwen3-8B  ~16 GB BF16 on disk
# Draft model for Method 3 (draft_target)
echo "[2/3] Downloading Qwen/Qwen3-8B  (~16 GB)..."
uv run hf download Qwen/Qwen3-8B
echo "      Done."
echo ""

# ── Model 3: Eagle3 speculator head for Method 4 ────────────────────────────
# AngelSlim/Qwen3-32B_eagle3  ~2 GB (head weights only)
# Eagle3 head for Qwen3-32B dense, 1.66-1.85x speedup on H20 (confirmed by AngelSlim)
echo "[3/3] Downloading AngelSlim/Qwen3-32B_eagle3  (~2 GB)..."
uv run hf download AngelSlim/Qwen3-32B_eagle3
echo "      Done."
echo ""

# ── All done, print next steps ─────────────────────────────────────────────
echo "=== All models downloaded (~84 GB total). ==="
echo ""
echo "Next: launch the container (Terminal 1) and start the server:"
echo ""
echo "  docker run --gpus all --ipc=host --name vllm-benchmark \\"
echo "    -v \$(pwd):/workspace \\"
echo "    -v \${HF_HOME:-\$HOME/.cache/huggingface}:/root/.cache/huggingface \\"
echo "    -p 8000:8000 \\"
echo "    nvcr.io/nvidia/vllm:26.04-py3 \\"
echo "    bash /workspace/start_server.sh baseline"
echo ""
echo "Then open Terminal 2 and run the benchmark client:"
echo ""
echo "  docker exec -it vllm-benchmark bash"
echo "  python /workspace/src/run_client.py --method baseline"
echo ""
echo "See README.md for the full two-terminal workflow."
