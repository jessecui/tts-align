"""Composite reward function.

This is the single entrypoint used by:
    - Offline dataset scoring (Phase 2): score each candidate, store in parquet.
    - Online GRPO reward computation (Phase 4): wrapped to take token ids and
      a reference DAC decoder, then forward through this same function.

Composite formula (default weights, configurable via constructor):
    composite = w_wer  * (1 - clamp(wer, 0, 1))
              + w_utmos * normalize_utmos(utmos)
              + w_spk   * 0.5 * (speaker_sim + 1)   # cosine in [-1,1] -> [0,1]

Notes:
    - WER is clamped to [0, 1] before inversion. A WER of 1.5 (more insertions
      than reference words) is "as bad as" a WER of 1.0 for ranking purposes.
    - speaker_sim contribution is optional; when no reference is provided, the
      weight is redistributed across WER and UTMOS proportionally so the
      composite stays on [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .utmos import compute_utmos, normalize_utmos
from .whisper_wer import compute_wer


@dataclass
class CompositeWeights:
    wer: float = 0.6
    utmos: float = 0.4
    speaker_sim: float = 0.0  # off by default; set to 0.2-0.3 for voice-cloning runs


@dataclass
class RewardConfig:
    weights: CompositeWeights = field(default_factory=CompositeWeights)
    whisper_model: str = "small"
    whisper_language: str = "en"


def score(
    audio: np.ndarray,
    target_text: str,
    sample_rate: int,
    reference_audio: Optional[np.ndarray] = None,
    reference_sr: Optional[int] = None,
    config: Optional[RewardConfig] = None,
) -> dict:
    """Score one generated audio sample. Returns dict with all sub-metrics + composite.

    Args:
        audio: 1-D float32 mono array of the generated speech.
        target_text: the ground-truth transcript we're trying to synthesize.
        sample_rate: sample rate of `audio`.
        reference_audio: optional reference speaker clip for speaker_sim.
        reference_sr: sample rate of `reference_audio`; required if reference given.
        config: reward configuration; defaults sensibly if omitted.

    Returns:
        {
            "wer": float,                  # 0.0 = perfect transcription
            "utmos": float,                # raw MOS, ~[1, 5]
            "speaker_sim": Optional[float],# cosine in [-1, 1], or None
            "composite": float,            # weighted, in [0, 1]
        }
    """
    cfg = config or RewardConfig()
    w = cfg.weights

    wer = compute_wer(audio, target_text, sample_rate, model_name=cfg.whisper_model, language=cfg.whisper_language)
    utmos = compute_utmos(audio, sample_rate)

    spk_sim: Optional[float] = None
    if reference_audio is not None:
        if reference_sr is None:
            raise ValueError("reference_audio provided but reference_sr is None")
        # Import lazily so non-voice-cloning runs don't pull in SpeechBrain.
        from .speaker_sim import compute_speaker_sim

        spk_sim = compute_speaker_sim(audio, sample_rate, reference_audio, reference_sr)

    wer_component = 1.0 - max(0.0, min(1.0, wer))
    utmos_component = normalize_utmos(utmos)

    if spk_sim is not None and w.speaker_sim > 0:
        spk_component = 0.5 * (spk_sim + 1.0)  # [-1, 1] -> [0, 1]
        total_w = w.wer + w.utmos + w.speaker_sim
        composite = (w.wer * wer_component + w.utmos * utmos_component + w.speaker_sim * spk_component) / total_w
    else:
        # Renormalize when speaker_sim is unused, so the composite still spans [0, 1].
        total_w = w.wer + w.utmos
        composite = (w.wer * wer_component + w.utmos * utmos_component) / total_w

    return {
        "wer": wer,
        "utmos": utmos,
        "speaker_sim": spk_sim,
        "composite": float(composite),
    }
