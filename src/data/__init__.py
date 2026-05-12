"""Dataset loading + preference-pair construction.

Public API:
    load_scored_dataset(path) -> DataFrame
    build_dpo_pairs(df, split="train", drop_ties=True) -> DataFrame
    build_kto_labels(df, split="train", desirable_threshold=0.7, undesirable_threshold=0.4) -> DataFrame
"""
from .dataset import build_dpo_pairs, build_kto_labels, load_scored_dataset

__all__ = ["load_scored_dataset", "build_dpo_pairs", "build_kto_labels"]
