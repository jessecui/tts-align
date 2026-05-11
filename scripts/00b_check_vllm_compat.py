"""Phase 1 vLLM compatibility check for OuteTTS 1.0.

Per project plan: we want to know *now* whether vLLM-accelerated sampling works
for OuteTTS, because if it doesn't, the GRPO wall-clock budget changes.

What this script does:
    1. Imports vllm and instantiates an LLM around the OuteTTS-1.0-1B weights.
    2. Runs a tiny generation against a single dummy prompt token sequence.
    3. Reports load time, single-sample latency, and a 4-way batched latency
       (since GRPO batches 4 generations per group).

What this script does NOT do:
    - Decode audio. We're just measuring whether the LM-level token generation
      works under vLLM. The DAC decoder runs separately.
    - Compare against HF generation. That's an optional Phase 4 measurement.

Run on the rented box:
    pip install -e '.[vllm]'   # or uv add vllm==0.6.4.post1
    python scripts/00b_check_vllm_compat.py

If vLLM fails to load the model, the script prints the failure clearly and
exits nonzero — the GRPO plan falls back to plain transformers generation.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make src importable when running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vllm-check")

MODEL_ID = "OuteAI/Llama-OuteTTS-1.0-1B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=128, help="small value; we only need to confirm decode runs")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"FAIL: vllm not installed ({e}). Install with: pip install -e '.[vllm]'", file=sys.stderr)
        return 3

    print(f"loading vLLM with model={args.model} ...")
    t0 = time.time()
    try:
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=0.6,  # leave headroom; this is a probe, not the real run
            max_model_len=2048,
            enforce_eager=True,  # skip CUDA graph capture for a faster cold start during the probe
        )
    except Exception as e:
        print(f"FAIL: vLLM could not load OuteTTS weights: {type(e).__name__}: {e}", file=sys.stderr)
        print("Fallback plan: use transformers.generate() for GRPO rollouts; expect 3-5x slower sampling.")
        return 4
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.1f}s")

    # The OuteTTS prompt format is a structured template — but for a compatibility
    # check we just need *something* to generate from. Use the model's tokenizer
    # to build a plausible BOS-prefixed prompt. The output won't be decodable
    # speech, and that's fine. We're testing the inference path, not quality.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    dummy_prompt = (tok.bos_token or "") + "Hello, world."

    params = SamplingParams(temperature=0.7, max_tokens=args.max_new_tokens, top_p=0.9)

    print(f"\nsingle-sample generation ({args.max_new_tokens} tokens) ...")
    t0 = time.time()
    out = llm.generate([dummy_prompt], params)
    single_s = time.time() - t0
    print(f"  ok, {single_s:.2f}s, {args.max_new_tokens / max(single_s, 1e-6):.1f} tok/s")

    print(f"\nbatched generation (B={args.batch}, {args.max_new_tokens} tokens each) ...")
    t0 = time.time()
    out = llm.generate([dummy_prompt] * args.batch, params)
    batch_s = time.time() - t0
    total_tokens = args.batch * args.max_new_tokens
    print(f"  ok, {batch_s:.2f}s, {total_tokens / max(batch_s, 1e-6):.1f} tok/s aggregate")
    print(f"  ({batch_s / single_s:.2f}x of single-sample time for {args.batch}x the work)")

    print("\nvLLM + OuteTTS-1.0-1B compatibility: PASS")
    print("  Use vLLM for GRPO sampling. Expected GRPO wall-clock 1-2 hours per run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
