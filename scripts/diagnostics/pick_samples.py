"""Copy a curated set of audio samples into results/samples/ for the README.

Picks the 3 eval prompts where the BASE model had the highest WER (i.e., the
prompts where preference optimization had the most room to help), then copies
the base/DPO/KTO renditions of each so a listener can A/B them.

Output layout:
    results/samples/<prompt_id>/prompt.txt
    results/samples/<prompt_id>/base.wav
    results/samples/<prompt_id>/dpo.wav
    results/samples/<prompt_id>/kto.wav
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    eval_parquet = Path("results/eval.parquet")
    audio_dir = Path("results/audio")
    out_root = Path("results/samples")

    if not eval_parquet.exists():
        print(f"missing {eval_parquet}; run ./run.sh eval first", file=sys.stderr)
        return 1

    df = pd.read_parquet(eval_parquet)
    # Top-3 hardest prompts by base WER (ties broken by composite ascending).
    base = df[df["method"] == "base"].sort_values(["wer", "composite"], ascending=[False, True]).head(3)
    print(f"Picked {len(base)} prompts (highest base WER):")
    for _, row in base.iterrows():
        print(f"  {row['prompt_id']}  base WER={row['wer']:.3f}  composite={row['composite']:.3f}")
        print(f"    text: {row['prompt']}")

    for _, row in base.iterrows():
        pid = row["prompt_id"]
        out_dir = out_root / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt.txt").write_text(row["prompt"] + "\n", encoding="utf-8")
        for method in ["base", "dpo", "kto"]:
            src = audio_dir / method / f"{pid}.wav"
            dst = out_dir / f"{method}.wav"
            if not src.exists():
                print(f"  WARN: missing {src}", file=sys.stderr)
                continue
            shutil.copy(src, dst)

    print(f"\nWrote samples to {out_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
