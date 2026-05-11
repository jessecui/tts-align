"""Reward pipeline: WER (Whisper), UTMOS (predicted MOS), speaker similarity, composite.

Public API:
    score(audio, target_text, sample_rate, reference_audio=None, reference_sr=None, config=None) -> dict
    RewardConfig, CompositeWeights
"""
from .composite import CompositeWeights, RewardConfig, score

__all__ = ["score", "RewardConfig", "CompositeWeights"]
