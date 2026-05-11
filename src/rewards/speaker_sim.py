"""Speaker similarity via ECAPA-TDNN embeddings.

We embed both the generated audio and a reference speaker clip with the
SpeechBrain ECAPA-TDNN model (trained for speaker verification on VoxCeleb)
and return the cosine similarity of the two embeddings.

Why ECAPA-TDNN instead of WavLM-base-plus-sv:
    - Single ~17 MB checkpoint, no big transformer to load.
    - SpeechBrain wrapper is stable and well-documented.
    - Verification-trained → produces embeddings explicitly tuned for
      identity discrimination, which is what we want.
    - If we later want a WavLM variant, the public interface here (a function
      that takes two waveforms and returns a similarity float) doesn't change.

Returned similarity is the raw cosine in [-1, 1]. We do NOT clamp to [0, 1] in
this module — the composite scorer decides how to map it onto the same axis
as WER/UTMOS.
"""
from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ECAPA_NATIVE_SR = 16_000  # the SR the checkpoint was trained on


@functools.lru_cache(maxsize=1)
def _load_model(device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Import inside the loader so module import doesn't pull in SpeechBrain at top level.
    from speechbrain.inference.speaker import EncoderClassifier

    logger.info("loading ECAPA-TDNN speaker encoder on device=%s", device)
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        # Cache under ./hf_cache (gitignored) instead of the SpeechBrain default
        # so the rented box's persistent volume can hold it.
        savedir="hf_cache/spkrec-ecapa-voxceleb",
    )


def _to_16k_mono(audio: np.ndarray, sample_rate: int) -> torch.Tensor:
    if audio.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {audio.shape}")
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    t = torch.from_numpy(audio)
    if sample_rate != ECAPA_NATIVE_SR:
        import torchaudio.functional as TF

        t = TF.resample(t.unsqueeze(0), orig_freq=sample_rate, new_freq=ECAPA_NATIVE_SR).squeeze(0)
    return t


def _embed(audio: np.ndarray, sample_rate: int) -> torch.Tensor:
    model = _load_model()
    wav = _to_16k_mono(audio, sample_rate)
    # SpeechBrain expects shape (batch, time). encode_batch returns (batch, 1, dim).
    with torch.no_grad():
        emb = model.encode_batch(wav.unsqueeze(0))
    return emb.squeeze(0).squeeze(0)  # (dim,)


def compute_speaker_sim(
    generated_audio: np.ndarray,
    generated_sr: int,
    reference_audio: np.ndarray,
    reference_sr: int,
) -> float:
    """Cosine similarity between the generated audio and the reference speaker clip.

    Returns:
        A float in [-1, 1]. Higher means closer voice. For OuteTTS voice cloning
        we expect values around 0.5-0.85 for in-distribution targets.
    """
    e_gen = _embed(generated_audio, generated_sr)
    e_ref = _embed(reference_audio, reference_sr)
    return float(F.cosine_similarity(e_gen.unsqueeze(0), e_ref.unsqueeze(0)).item())
