"""Smoke test for the reward pipeline.

Generates a handful of audio samples with the base OuteTTS model, scores each
one, prints per-sample results, and prints min/max/mean per metric across the
batch. Useful before generating the full dataset to confirm the reward stack
is wired up correctly.

Run on the rented box:
    ./run.sh smoke
    ./run.sh smoke --voice-cloning           # also test speaker_sim
    ./run.sh smoke --skip-generation --audio-dir <path>

What we're sanity-checking:
    - The base model installs and runs.
    - WER, UTMOS, (optional) speaker_sim all return numbers in expected ranges.
    - The metrics discriminate: across a few samples we should see *some*
      variation in WER and UTMOS. If every sample scores identically, something
      is wired wrong (e.g. constant-output bug, or the model isn't actually
      decoding).
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

# Make src importable when running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rewards import RewardConfig, CompositeWeights, score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke")


SMOKE_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Please call Stella at four-fifteen on Tuesday.",
    "She sells seashells by the seashore on a sunny afternoon.",
    "The capital of Australia is Canberra, not Sydney.",
    "In nineteen sixty-nine, humans first walked on the moon.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize_outetts(prompts: list[str], out_dir: Path, voice_cloning: bool) -> list[tuple[Path, str, Optional[Path]]]:
    """Synthesize one audio file per prompt with OuteTTS 1.0 (Llama 1B variant).

    Returns a list of (audio_path, target_text, reference_audio_path_or_None).

    Uses the high-level outetts.Interface — we only need the audio out, not raw
    tokens, at this phase. The training scripts will load the underlying Llama
    model directly via transformers.
    """
    import outetts

    logger.info("loading OuteTTS-1.0-1B (HF backend) — first run will download ~2GB")
    iface = outetts.Interface(
        config=outetts.ModelConfig.auto_config(
            model=outetts.Models.VERSION_1_0_SIZE_1B,
            backend=outetts.Backend.HF,
        )
    )

    # Pick a built-in speaker. For voice-cloning mode, we'll use this same
    # speaker's audio as the reference for speaker_sim.
    speaker = iface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    ref_path: Optional[Path] = None
    if voice_cloning:
        # The Interface exposes the loaded speaker's reference clip in different
        # ways across releases (speaker.audio / speaker.wav / speaker["audio"]).
        # Rather than reach in, we synthesize one short reference clip with this
        # speaker reading a fixed sentence, and reuse it as the speaker_sim
        # anchor. Smoke test only — circular but fine for verifying the metric runs.
        ref_path = out_dir / "reference_speaker.wav"
        if not ref_path.exists():
            logger.info("synthesizing reference speaker clip -> %s", ref_path)
            ref_out = iface.generate(
                config=outetts.GenerationConfig(
                    text="This is a reference clip for the speaker similarity metric.",
                    speaker=speaker,
                    sampler_config=outetts.SamplerConfig(temperature=0.4),
                )
            )
            ref_out.save(str(ref_path))

    items: list[tuple[Path, str, Optional[Path]]] = []
    for i, prompt in enumerate(prompts):
        path = out_dir / f"sample_{i:02d}.wav"
        logger.info("[%d/%d] synthesizing: %r", i + 1, len(prompts), prompt[:60])
        t0 = time.time()
        out = iface.generate(
            config=outetts.GenerationConfig(
                text=prompt,
                speaker=speaker,
                sampler_config=outetts.SamplerConfig(temperature=0.7),
            )
        )
        out.save(str(path))
        logger.info("    saved -> %s (%.1fs)", path, time.time() - t0)
        items.append((path, prompt, ref_path))
    return items


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # to mono
    return audio.astype(np.float32), int(sr)


def print_distribution(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label}: (no values)")
        return
    print(f"  {label}: min={min(values):.4f}  max={max(values):.4f}  mean={statistics.mean(values):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("data/smoke"), help="where to write/read smoke audio")
    parser.add_argument("--skip-generation", action="store_true", help="skip synthesis; score existing files in --audio-dir")
    parser.add_argument("--voice-cloning", action="store_true", help="include speaker_sim metric")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whisper-model", default="small")
    args = parser.parse_args()

    set_seed(args.seed)
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    # ----- Generate (or reuse existing) samples -----
    if args.skip_generation:
        wavs = sorted(p for p in args.audio_dir.glob("sample_*.wav"))
        assert wavs, f"no sample_*.wav found in {args.audio_dir}"
        # When skipping generation, we don't have the original prompts paired
        # with files. Require a sidecar prompts.txt; otherwise reuse SMOKE_PROMPTS
        # by index.
        prompts_file = args.audio_dir / "prompts.txt"
        if prompts_file.exists():
            prompts = [ln.strip() for ln in prompts_file.read_text().splitlines() if ln.strip()]
        else:
            prompts = SMOKE_PROMPTS[: len(wavs)]
            assert len(prompts) == len(wavs), "prompt/file count mismatch; provide prompts.txt"
        ref_path = (args.audio_dir / "reference_speaker.wav") if args.voice_cloning else None
        if ref_path and not ref_path.exists():
            ref_path = None
        items = [(w, p, ref_path) for w, p in zip(wavs, prompts)]
    else:
        items = synthesize_outetts(SMOKE_PROMPTS, args.audio_dir, voice_cloning=args.voice_cloning)
        (args.audio_dir / "prompts.txt").write_text("\n".join(SMOKE_PROMPTS) + "\n")

    # ----- Score -----
    weights = CompositeWeights(
        wer=0.6,
        utmos=0.4,
        speaker_sim=0.3 if args.voice_cloning else 0.0,
    )
    cfg = RewardConfig(weights=weights, whisper_model=args.whisper_model)

    results: list[dict] = []
    print("\n=== per-sample scores ===")
    for path, prompt, ref in items:
        audio, sr = load_audio(path)
        ref_audio, ref_sr = (load_audio(ref) if ref is not None else (None, None))
        t0 = time.time()
        out = score(
            audio=audio,
            target_text=prompt,
            sample_rate=sr,
            reference_audio=ref_audio,
            reference_sr=ref_sr,
            config=cfg,
        )
        elapsed = time.time() - t0
        results.append(out)
        spk = f" spk={out['speaker_sim']:.3f}" if out["speaker_sim"] is not None else ""
        print(f"  {path.name}  wer={out['wer']:.3f}  utmos={out['utmos']:.2f}{spk}  composite={out['composite']:.3f}  ({elapsed:.1f}s)")

    # ----- Distribution -----
    print("\n=== distribution across samples ===")
    print_distribution("wer        ", [r["wer"] for r in results])
    print_distribution("utmos      ", [r["utmos"] for r in results])
    if args.voice_cloning:
        print_distribution("speaker_sim", [r["speaker_sim"] for r in results if r["speaker_sim"] is not None])
    print_distribution("composite  ", [r["composite"] for r in results])

    # ----- Sanity assertions -----
    # WER should land in [0, 2] for any plausibly-functioning TTS (the upper end
    # leaves room for catastrophic failures with many insertions).
    # UTMOS should be in [1, 5] by construction.
    # Composite should be in [0, 1].
    for r in results:
        assert 0.0 <= r["wer"] <= 2.0, f"wer out of range: {r['wer']}"
        assert 1.0 <= r["utmos"] <= 5.0, f"utmos out of range: {r['utmos']}"
        assert 0.0 <= r["composite"] <= 1.0, f"composite out of range: {r['composite']}"

    # Discrimination check: if every sample produced the exact same WER or UTMOS,
    # something is suspicious. With ~5 samples we expect at least some spread.
    wer_spread = max(r["wer"] for r in results) - min(r["wer"] for r in results)
    utmos_spread = max(r["utmos"] for r in results) - min(r["utmos"] for r in results)
    if wer_spread < 1e-4 and utmos_spread < 1e-4:
        print("\nWARNING: zero spread in both WER and UTMOS — metrics may not be discriminating.")
        return 2

    print("\nsmoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
