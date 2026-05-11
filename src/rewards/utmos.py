"""Predicted MOS via UTMOS.

UTMOS predicts a Mean Opinion Score for naturalness on the 1-5 MOS scale
(synth speech datasets typically score 2-4.5). We use the fakerybakery/utmos
pip package — an unofficial but maintained wrapper around the original
SaruLab UTMOS model.

Outputs are not normalized here; the composite scorer handles min-max scaling.
Caller can divide by 5.0 if they want a [0, 1] proxy in isolation.
"""
from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np
import torch
import utmos as _utmos_pkg

logger = logging.getLogger(__name__)

UTMOS_MIN, UTMOS_MAX = 1.0, 5.0  # the rating scale UTMOS was trained on


@functools.lru_cache(maxsize=1)
def _load_model(device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading UTMOS model on device=%s", device)
    # The fakerybakery utmos.Score class auto-detects device.
    return _utmos_pkg.Score()


def compute_utmos(audio: np.ndarray, sample_rate: int) -> float:
    """Predict MOS for the given audio array.

    Args:
        audio: 1-D float32 mono array.
        sample_rate: sample rate of `audio`. UTMOS resamples internally.

    Returns:
        Predicted MOS in roughly [1, 5]. Higher is better.
    """
    if audio.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {audio.shape}")
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    model = _load_model()
    # The package accepts a torch.Tensor or np.ndarray (sample_len,) or (B, sample_len).
    score = model.calculate_wav(torch.from_numpy(audio), sample_rate)
    # Some versions return a tensor; some return a python float. Coerce.
    if isinstance(score, torch.Tensor):
        score = score.item()
    return float(score)


def normalize_utmos(mos: float) -> float:
    """Min-max scale a raw MOS into [0, 1] using the trained range.

    Used by the composite scorer so UTMOS and (1 - WER) live on the same scale.
    Clamps out-of-range values rather than failing — UTMOS occasionally predicts
    slightly outside [1, 5] on degenerate inputs.
    """
    clamped = max(UTMOS_MIN, min(UTMOS_MAX, mos))
    return (clamped - UTMOS_MIN) / (UTMOS_MAX - UTMOS_MIN)
