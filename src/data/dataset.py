"""Load the scored preference parquet and build chosen/rejected pairs.

Same parquet feeds DPO (paired) and KTO (binary-thresholded). This module owns
both the loading and the pair-construction logic so the training scripts stay
small and the rules live in one place.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"prompt_id", "prompt", "split", "candidate_id", "audio_path", "composite"}


def load_scored_dataset(parquet_path: Path) -> pd.DataFrame:
    """Read the scored parquet and sanity-check its schema."""
    df = pd.read_parquet(parquet_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"dataset {parquet_path} missing columns: {missing}")
    return df


def build_dpo_pairs(
    df: pd.DataFrame,
    split: str = "train",
    top_k: int = 2,
    bottom_k: int = 2,
    drop_ties: bool = True,
) -> pd.DataFrame:
    """For each prompt, cross-pair top-K candidates with bottom-K candidates.

    With K=2 and 4 candidates per prompt, this yields top-2 x bottom-2 = 4 pairs
    per prompt (212 pairs from 53 train prompts in this project's dataset).
    Parallels KTO's quantile logic (top-half preferred over bottom-half) and
    gives DPO comparable training data quantity to KTO so a DPO-vs-KTO
    comparison is apples-to-apples on data quantity.

    Earlier versions of this function only paired the single best vs the single
    worst per prompt (1 pair/prompt = 53 pairs). That was data-starved relative
    to KTO and made the loss-vs-data-quantity attribution ambiguous in the
    eval comparison. Top-K vs bottom-K resolves that.

    Returns columns:
        prompt_id, prompt, chosen_audio_path, rejected_audio_path,
        chosen_composite, rejected_composite, margin.
    """
    df = df[df["split"] == split]
    pairs = []
    for prompt_id, group in df.groupby("prompt_id"):
        if len(group) < top_k + bottom_k:
            continue
        srt = group.sort_values("composite", ascending=False)
        top = srt.iloc[:top_k]
        bottom = srt.iloc[-bottom_k:]
        for _, chosen in top.iterrows():
            for _, rejected in bottom.iterrows():
                if drop_ties and chosen["composite"] == rejected["composite"]:
                    continue
                pairs.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt": chosen["prompt"],
                        "chosen_audio_path": chosen["audio_path"],
                        "rejected_audio_path": rejected["audio_path"],
                        "chosen_composite": float(chosen["composite"]),
                        "rejected_composite": float(rejected["composite"]),
                        "margin": float(chosen["composite"] - rejected["composite"]),
                    }
                )
    return pd.DataFrame(pairs)


def build_kto_labels(
    df: pd.DataFrame,
    split: str = "train",
    desirable_threshold: float = 0.7,
    undesirable_threshold: float = 0.4,
) -> pd.DataFrame:
    """For KTO: per-candidate binary label. composite > desirable_threshold -> desirable,
    composite < undesirable_threshold -> undesirable, in between -> dropped.

    Returns columns: prompt_id, prompt, audio_path, composite, desirable (bool).
    """
    df = df[df["split"] == split].copy()
    keep = (df["composite"] > desirable_threshold) | (df["composite"] < undesirable_threshold)
    out = df[keep].copy()
    out["desirable"] = out["composite"] > desirable_threshold
    return out[["prompt_id", "prompt", "audio_path", "composite", "desirable"]].reset_index(drop=True)
