"""Generate the scored preference dataset for DPO/KTO/GRPO.

Pipeline per prompt:
    1. Read prompt text (easy from data/easy_prompts.txt, hard from data/hard_prompts.txt).
    2. Synthesize N candidates with OuteTTS at varied sampling temperatures.
    3. Score every candidate with the reward pipeline (WER, UTMOS, optional speaker_sim).
    4. Save audio to data/generated/<prompt_id>_<cand_id>.wav and rows to a parquet.

Resumable: re-running the script picks up where it left off based on what's
already in the parquet.

Defaults (override via `--n-easy`, `--n-hard`, `--n-eval`):
    up to 100 easy + 50 hard prompts (capped by what's in the bundled prompt files),
    4 candidates each at temps [0.7, 0.9, 1.0, 1.2], 30 held out as eval.
    Roughly 2-3 hours on an A100 40GB at default scale.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rewards import CompositeWeights, RewardConfig, score  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dataset")


DEFAULT_TEMPS = [0.7, 0.9, 1.0, 1.2]
SMOKE_TEMPS = [0.7, 1.2]


def read_prompts_file(path: Path) -> list[str]:
    """Read non-empty, non-comment lines from a prompts text file."""
    lines = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            lines.append(s)
    return lines


def build_prompt_records(
    easy_prompts: list[str],
    hard_prompts: list[str],
    n_easy: int,
    n_hard: int,
    n_eval: int,
    seed: int,
) -> list[dict]:
    """Combine easy + hard, assign train/eval split by-prompt (deterministic given seed).

    Eval set is drawn proportionally from both pools so the eval distribution looks
    like the train distribution.
    """
    rng = np.random.default_rng(seed)
    easy = list(rng.permutation(easy_prompts))[:n_easy]
    hard = list(rng.permutation(hard_prompts))[:n_hard]
    if len(easy) < n_easy:
        logger.warning("only %d easy prompts available (requested %d)", len(easy), n_easy)
    if len(hard) < n_hard:
        logger.warning("only %d hard prompts available (requested %d)", len(hard), n_hard)

    # Proportional eval split across pools
    n_eval_easy = round(n_eval * len(easy) / max(1, len(easy) + len(hard)))
    n_eval_hard = n_eval - n_eval_easy

    records = []
    for i, p in enumerate(easy):
        records.append({"prompt": p, "pool": "easy", "split": "eval" if i < n_eval_easy else "train"})
    for i, p in enumerate(hard):
        records.append({"prompt": p, "pool": "hard", "split": "eval" if i < n_eval_hard else "train"})

    # Assign stable prompt IDs after shuffling so we don't depend on file order
    rng.shuffle(records)
    for i, r in enumerate(records):
        r["prompt_id"] = f"p{i:04d}"
    return records


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def write_partial(rows: list[dict], existing: pd.DataFrame, out_path: Path) -> None:
    """Append rows to the parquet on disk. Called periodically so a crash doesn't lose progress."""
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if rows else existing
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easy-prompts", type=Path, default=Path("data/easy_prompts.txt"))
    parser.add_argument("--hard-prompts", type=Path, default=Path("data/hard_prompts.txt"))
    parser.add_argument("--n-easy", type=int, default=100)
    parser.add_argument("--n-hard", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=30, help="prompts held out for eval")
    parser.add_argument("--audio-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset.parquet"))
    parser.add_argument("--no-voice-cloning", action="store_true", help="skip speaker_sim metric")
    parser.add_argument("--reference-audio", type=Path, default=None,
                        help="external reference clip for speaker_sim (overrides the synthesized one)")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10, help="flush parquet every N new rows")
    parser.add_argument("--smoke-test", action="store_true",
                        help="5 prompts, 2 temperatures, no resume — for quick pipeline check")
    args = parser.parse_args()

    set_seed(args.seed)
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    # ----- Prompts -----
    easy = read_prompts_file(args.easy_prompts)
    hard = read_prompts_file(args.hard_prompts)

    if args.smoke_test:
        records = [
            {"prompt_id": f"smoke{i:02d}", "prompt": p, "pool": "easy", "split": "train"}
            for i, p in enumerate((easy + hard)[:5])
        ]
        temps = SMOKE_TEMPS
        args.output = args.output.with_name("dataset_smoke.parquet")
        args.audio_dir = args.audio_dir / "smoke"
        args.audio_dir.mkdir(parents=True, exist_ok=True)
    else:
        records = build_prompt_records(easy, hard, args.n_easy, args.n_hard, args.n_eval, args.seed)
        temps = DEFAULT_TEMPS

    logger.info("prompts: %d total (%d train, %d eval) x %d temperatures = %d candidates",
                len(records),
                sum(r["split"] == "train" for r in records),
                sum(r["split"] == "eval" for r in records),
                len(temps),
                len(records) * len(temps))

    # ----- Resume -----
    if args.output.exists() and not args.smoke_test:
        existing = pd.read_parquet(args.output)
        done = set(zip(existing["prompt_id"], existing["candidate_id"]))
        logger.info("resuming: %d candidates already scored", len(done))
    else:
        existing = pd.DataFrame()
        done = set()

    # ----- TTS -----
    logger.info("loading OuteTTS-1.0-1B (HF backend)")
    import outetts  # local import; outetts pulls heavy deps at import time

    iface = outetts.Interface(
        config=outetts.ModelConfig.auto_config(
            model=outetts.Models.VERSION_1_0_SIZE_1B,
            backend=outetts.Backend.HF,
        )
    )
    speaker = iface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    # Reference clip for speaker_sim. If the user didn't pass one, synthesize a short
    # one with the chosen speaker; this anchors the cosine to "the voice OuteTTS is
    # trying to imitate" which is the most coherent target for a voice-cloning eval.
    ref_path: Path | None = None
    if not args.no_voice_cloning:
        if args.reference_audio is not None:
            ref_path = args.reference_audio
        else:
            ref_path = args.audio_dir / "reference_speaker.wav"
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

    ref_audio_array, ref_sr = (None, None)
    if ref_path is not None:
        ref_audio_array, ref_sr = load_audio(ref_path)

    # ----- Reward config -----
    weights = CompositeWeights(
        wer=0.6,
        utmos=0.4,
        speaker_sim=0.0 if args.no_voice_cloning else 0.3,
    )
    reward_cfg = RewardConfig(weights=weights, whisper_model=args.whisper_model)

    # ----- Generate + score -----
    new_rows: list[dict] = []
    total = len(records) * len(temps)
    counter = 0
    t_start = time.time()

    try:
        for record in records:
            for tid, temp in enumerate(temps):
                counter += 1
                cand_id = f"t{tid}"
                if (record["prompt_id"], cand_id) in done:
                    continue
                audio_path = args.audio_dir / f"{record['prompt_id']}_{cand_id}.wav"

                t0 = time.time()
                try:
                    out = iface.generate(
                        config=outetts.GenerationConfig(
                            text=record["prompt"],
                            speaker=speaker,
                            sampler_config=outetts.SamplerConfig(temperature=temp),
                        )
                    )
                    out.save(str(audio_path))
                    audio, sr = load_audio(audio_path)
                    scores = score(
                        audio=audio,
                        target_text=record["prompt"],
                        sample_rate=sr,
                        reference_audio=ref_audio_array,
                        reference_sr=ref_sr,
                        config=reward_cfg,
                    )
                except Exception as e:
                    logger.exception("[%d/%d] failed on %s/%s: %s", counter, total,
                                     record["prompt_id"], cand_id, e)
                    continue

                elapsed = time.time() - t0
                row = {
                    **record,
                    "candidate_id": cand_id,
                    "temperature": temp,
                    "audio_path": str(audio_path),
                    **scores,
                    "elapsed_s": elapsed,
                }
                new_rows.append(row)
                spk = f" spk={scores['speaker_sim']:.3f}" if scores["speaker_sim"] is not None else ""
                logger.info("[%d/%d] %s/%s T=%.2f wer=%.3f utmos=%.2f%s composite=%.3f (%.1fs)",
                            counter, total, record["prompt_id"], cand_id, temp,
                            scores["wer"], scores["utmos"], spk, scores["composite"], elapsed)

                if len(new_rows) % args.save_every == 0:
                    write_partial(new_rows, existing, args.output)

    finally:
        # Always flush so an interrupted run leaves a valid partial parquet behind.
        write_partial(new_rows, existing, args.output)

    total_rows = len(existing) + len(new_rows)
    total_time = time.time() - t_start
    logger.info("done. %d new rows (%d total). %.1f min. -> %s",
                len(new_rows), total_rows, total_time / 60, args.output)

    # ----- Quick distribution summary -----
    final = pd.read_parquet(args.output)
    print("\n=== dataset summary ===")
    print(f"  rows: {len(final)}")
    print(f"  prompts: {final['prompt_id'].nunique()}")
    print(f"  splits: {dict(final.groupby('split').size())}")
    for col in ("wer", "utmos", "composite"):
        s = final[col].astype(float)
        print(f"  {col:>10}: min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}  std={s.std():.3f}")
    if "speaker_sim" in final.columns and final["speaker_sim"].notna().any():
        s = final["speaker_sim"].dropna().astype(float)
        print(f"  spk_sim   : min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}  std={s.std():.3f}")
    # How much spread per prompt? (margin between best and worst candidate)
    margins = final.groupby("prompt_id")["composite"].agg(lambda x: x.max() - x.min())
    print(f"  composite margin per prompt: min={margins.min():.3f}  mean={margins.mean():.3f}  max={margins.max():.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
