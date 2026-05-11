"""Word Error Rate from Whisper-small transcription.

WER = (S + D + I) / N, where S/D/I are substitution/deletion/insertion edit
operations and N is the number of reference words. A perfect transcription
scores 0; pathological outputs can exceed 1.0 (more insertions than the
reference length).
"""
from __future__ import annotations

import functools
import logging
from typing import Optional

import jiwer
import numpy as np
import torch
import whisper

logger = logging.getLogger(__name__)

WHISPER_NATIVE_SR = 16_000  # whisper resamples to 16k internally; expose for callers

# jiwer normalization pipeline: applied identically to hypothesis and reference
# before edit-distance scoring. Removes punctuation, lowercases, collapses
# whitespace, and strips empty strings. We intentionally keep contractions
# intact ("don't" stays "don't") to match Whisper's typical output.
_WER_NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


@functools.lru_cache(maxsize=1)
def _load_model(model_name: str = "small", device: Optional[str] = None) -> whisper.Whisper:
    """Load and cache the Whisper model. Singleton across calls."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("loading Whisper model=%s on device=%s", model_name, device)
    model = whisper.load_model(model_name, device=device)
    return model


def compute_wer(
    audio: np.ndarray,
    target_text: str,
    sample_rate: int,
    model_name: str = "small",
    language: str = "en",
) -> float:
    """Transcribe `audio` with Whisper and return WER against `target_text`.

    Args:
        audio: 1-D float32 array in [-1, 1]. Mono.
        target_text: ground-truth transcript.
        sample_rate: sample rate of `audio`. Resampled to 16k inside whisper.
        model_name: any whisper checkpoint ("tiny", "base", "small", ...).
        language: ISO-639-1 code; pinning avoids per-sample language detection cost.

    Returns:
        WER as a float. 0.0 == perfect. Can exceed 1.0.
    """
    if audio.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {audio.shape}")
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if sample_rate != WHISPER_NATIVE_SR:
        # whisper.transcribe expects either a path or a 16k float32 array.
        # Resample with torchaudio for accuracy parity with whisper's own loader.
        import torchaudio.functional as F  # local import; heavy

        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = F.resample(audio_t, orig_freq=sample_rate, new_freq=WHISPER_NATIVE_SR)
        audio = audio_t.squeeze(0).numpy()

    model = _load_model(model_name)
    # fp16=False on CPU; whisper auto-detects on GPU.
    result = model.transcribe(
        audio,
        language=language,
        fp16=(model.device.type == "cuda"),
        verbose=False,
    )
    hyp = result["text"].strip()

    # If both strings collapse to empty after normalization, jiwer raises.
    # An empty reference is a programming error; an empty hyp is a model failure
    # and should score as WER == 1.0 (all words deleted).
    ref_words = _WER_NORMALIZE(target_text)
    hyp_words = _WER_NORMALIZE(hyp)
    if not ref_words or not ref_words[0]:
        raise ValueError(f"empty reference after normalization: {target_text!r}")
    if not hyp_words or not hyp_words[0]:
        return 1.0

    return float(jiwer.wer(target_text, hyp, truth_transform=_WER_NORMALIZE, hypothesis_transform=_WER_NORMALIZE))
