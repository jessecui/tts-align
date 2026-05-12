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


def build_dpo_pairs(df: pd.DataFrame, split: str = "train", drop_ties: bool = True) -> pd.DataFrame:
    """For each prompt, pair the highest-composite candidate (chosen) with the lowest (rejected).

    Args:
        df: scored dataset (output of `load_scored_dataset`).
        split: which split column value to keep.
        drop_ties: if True, skip prompts where chosen and rejected have identical
            composite (no preference signal).

    Returns columns:
        prompt_id, prompt, chosen_audio_path, rejected_audio_path,
        chosen_composite, rejected_composite, margin.
    """
    df = df[df["split"] == split]
    pairs = []
    for prompt_id, group in df.groupby("prompt_id"):
        if len(group) < 2:
            continue
        srt = group.sort_values("composite", ascending=False)
        chosen, rejected = srt.iloc[0], srt.iloc[-1]
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
