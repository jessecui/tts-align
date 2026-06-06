"""DPO sequence-length audit.

Tokenizes a sample of DPO (prompt, chosen, rejected) triples and reports the
length distribution against the configured max_length / max_prompt_length
budget. The point is to know whether DPOTrainer is silently truncating any
chosen/rejected completions — which would corrupt the preference signal.

Runs on CPU by default (CUDA_VISIBLE_DEVICES hidden at the top) so it's safe to
run alongside an active GPU training job. Takes ~10-15 minutes on CPU.
"""
import os
# Hide the GPU before anything else imports torch — safety belt so this can
# run alongside an active GRPO training job without GPU contention.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import importlib.util
import statistics
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data import build_dpo_pairs, load_scored_dataset  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load(Path("config/dpo.yaml").read_text())
    df = load_scored_dataset(Path(cfg["dataset_path"]))
    pairs = build_dpo_pairs(df, split="train")
    print(f"total DPO pairs (train split): {len(pairs)}")

    import outetts

    print("loading OuteTTS interface on CPU...")
    iface = outetts.Interface(
        config=outetts.ModelConfig.auto_config(
            model=outetts.Models.VERSION_1_0_SIZE_1B,
            backend=outetts.Backend.HF,
        )
    )

    spec = importlib.util.spec_from_file_location("dpo_script", "scripts/02_train_dpo.py")
    dpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dpo)

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])

    n_sample = min(50, len(pairs))
    print(f"sampling {n_sample}/{len(pairs)} pairs")

    prompt_lens, chosen_lens, rejected_lens = [], [], []
    for i, (_, r) in enumerate(pairs.head(n_sample).iterrows()):
        p = dpo.build_outetts_prompt_str(iface, r["prompt"])
        c = dpo.encode_audio_to_dpo_completion(iface, Path(r["chosen_audio_path"]), r["prompt"])
        j = dpo.encode_audio_to_dpo_completion(iface, Path(r["rejected_audio_path"]), r["prompt"])
        prompt_lens.append(len(tok.encode(p, add_special_tokens=False)))
        chosen_lens.append(len(tok.encode(c, add_special_tokens=False)))
        rejected_lens.append(len(tok.encode(j, add_special_tokens=False)))
        if (i + 1) % 10 == 0:
            print(f"  encoded {i + 1}/{n_sample}")

    budget = cfg["max_length"] - cfg["max_prompt_length"]

    def stats(name: str, xs: list[int]) -> None:
        p90 = sorted(xs)[int(len(xs) * 0.9)]
        print(f"  {name}: min={min(xs)} median={statistics.median(xs):.0f} p90={p90} max={max(xs)}")

    print(f"\nsampled {len(prompt_lens)}/{len(pairs)} pairs; completion budget = {budget}")
    stats("prompt", prompt_lens)
    stats("chosen", chosen_lens)
    stats("rejected", rejected_lens)
    truncated = sum(1 for c, j in zip(chosen_lens, rejected_lens) if c > budget or j > budget)
    print(f"truncated pairs: {truncated}/{len(chosen_lens)} ({100 * truncated / len(chosen_lens):.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
