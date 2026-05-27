"""Phase 5: held-out comparison eval across {base, DPO, KTO}.

Pipeline:
    1. Load the held-out eval prompts (split=='eval' in dataset.parquet).
    2. For each method:
        - base: load the unmodified OuteTTS-1.0-1B.
        - DPO / KTO: merge the trained LoRA adapter into the base, save the
          merged model to disk, point outetts at that local path.
    3. Synthesize one audio file per (method, prompt) at a fixed temperature.
    4. Score each with the same reward pipeline used at training time
       (WER + UTMOS + optional speaker_sim + composite).
    5. Aggregate and print a comparison table; write a markdown summary for the README.

Output artifacts:
    results/audio/<method>/<prompt_id>.wav   — one audio per (method, prompt)
    results/eval.parquet                     — all per-sample scores
    results/comparison.md                    — summary table (paste into README)
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
logger = logging.getLogger("eval")


def _patch_whisper_load_model_cache() -> None:
    """Cache whisper.load_model so the reward pipeline doesn't reload Whisper for every sample."""
    import whisper

    _orig = whisper.load_model
    _cache: dict[tuple, object] = {}

    def _cached(name="small", device=None, *args, **kwargs):
        key = (name, str(device))
        if key not in _cache:
            _cache[key] = _orig(name, device=device, *args, **kwargs)
        return _cache[key]

    whisper.load_model = _cached


_patch_whisper_load_model_cache()


def find_latest_checkpoint(method_runs_dir: Path) -> Path:
    """Pick the latest non-smoke training run under runs/<method>/.

    Returns the run directory, which contains adapter_model.safetensors etc.
    """
    if not method_runs_dir.exists():
        raise FileNotFoundError(f"no runs directory at {method_runs_dir}")
    candidates = [
        d for d in method_runs_dir.iterdir()
        if d.is_dir() and "smoke" not in d.name and not d.name.endswith("-merged")
    ]
    if not candidates:
        raise FileNotFoundError(f"no non-smoke runs in {method_runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def merge_lora_to_dir(adapter_path: Path, base_model_id: str, output_dir: Path) -> None:
    """Merge a LoRA adapter into the base model and save it as a standalone HF model.

    Idempotent — skips work if `output_dir` already contains a model.
    """
    if output_dir.exists() and (output_dir / "config.json").exists():
        logger.info("merged model already at %s, skipping", output_dir)
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("merging LoRA from %s into base %s -> %s", adapter_path, base_model_id, output_dir)
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = peft_model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    AutoTokenizer.from_pretrained(base_model_id).save_pretrained(str(output_dir))
    del merged, peft_model, base
    torch.cuda.empty_cache()


def load_iface(model_path_override: str | None = None):
    """Load OuteTTS Interface. If `model_path_override` is given, point it at a local merged dir."""
    import outetts

    config = outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        backend=outetts.Backend.HF,
    )
    if model_path_override is not None:
        config.model_path = model_path_override
        config.tokenizer_path = model_path_override
    return outetts.Interface(config=config)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def evaluate_method(
    method_name: str,
    model_path_override: str | None,
    eval_prompts: list[tuple[str, str]],
    ref_audio: np.ndarray | None,
    ref_sr: int | None,
    audio_dir: Path,
    reward_cfg: RewardConfig,
    temperature: float = 0.7,
) -> list[dict]:
    """Synthesize + score every eval prompt with one method."""
    logger.info("=== evaluating: %s ===", method_name)
    import outetts

    iface = load_iface(model_path_override)
    speaker = iface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    out_dir = audio_dir / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, (pid, prompt_text) in enumerate(eval_prompts):
        audio_path = out_dir / f"{pid}.wav"
        t0 = time.time()
        try:
            out = iface.generate(config=outetts.GenerationConfig(
                text=prompt_text,
                speaker=speaker,
                sampler_config=outetts.SamplerConfig(temperature=temperature),
            ))
            out.save(str(audio_path))
            audio, sr = load_audio(audio_path)
            scores = score(
                audio=audio,
                target_text=prompt_text,
                sample_rate=sr,
                reference_audio=ref_audio,
                reference_sr=ref_sr,
                config=reward_cfg,
            )
        except Exception as e:
            logger.exception("[%s %d/%d] failed on %s: %s", method_name, i + 1, len(eval_prompts), pid, e)
            continue

        elapsed = time.time() - t0
        results.append({
            "method": method_name,
            "prompt_id": pid,
            "prompt": prompt_text,
            "audio_path": str(audio_path),
            **scores,
            "elapsed_s": elapsed,
        })
        spk = f" spk={scores['speaker_sim']:.3f}" if scores["speaker_sim"] is not None else ""
        logger.info(
            "[%s %d/%d] %s wer=%.3f utmos=%.2f%s composite=%.3f (%.1fs)",
            method_name, i + 1, len(eval_prompts), pid,
            scores["wer"], scores["utmos"], spk, scores["composite"], elapsed,
        )

    del iface
    torch.cuda.empty_cache()
    return results


def summarize(results_df: pd.DataFrame, methods: list[str], wer_failure_threshold: float = 0.30) -> pd.DataFrame:
    rows = []
    for method in methods:
        m = results_df[results_df["method"] == method]
        if len(m) == 0:
            continue
        row = {
            "method": method,
            "n": len(m),
            "mean_wer": float(m["wer"].mean()),
            "mean_utmos": float(m["utmos"].mean()),
            "mean_composite": float(m["composite"].mean()),
            "catastrophic_failure_rate": float((m["wer"] > wer_failure_threshold).mean()),
        }
        if "speaker_sim" in m.columns and m["speaker_sim"].notna().any():
            row["mean_speaker_sim"] = float(m["speaker_sim"].dropna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown_summary(summary_df: pd.DataFrame, n_eval: int, out_path: Path) -> None:
    has_spk = "mean_speaker_sim" in summary_df.columns and summary_df["mean_speaker_sim"].notna().any()
    md = [
        "# Eval Comparison\n",
        f"_{n_eval} held-out prompts. Same reward pipeline as training (Whisper-small WER, UTMOS, ECAPA-TDNN speaker sim, composite)._\n",
    ]
    header = "| method | n | mean WER ↓ | mean UTMOS ↑ |"
    sep = "|---|---|---|---|"
    if has_spk:
        header += " mean spk_sim ↑ |"
        sep += "---|"
    header += " mean composite ↑ | catastrophic failure (WER>30%) ↓ |"
    sep += "---|---|"
    md.append(header)
    md.append(sep)
    for _, r in summary_df.iterrows():
        line = f"| {r['method']} | {int(r['n'])} | {r['mean_wer']:.3f} | {r['mean_utmos']:.2f} |"
        if has_spk:
            spk = r.get("mean_speaker_sim", float("nan"))
            line += f" {spk:.3f} |"
        line += f" {r['mean_composite']:.3f} | {r['catastrophic_failure_rate'] * 100:.1f}% |"
        md.append(line)
    out_path.write_text("\n".join(md) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.parquet"))
    parser.add_argument("--audio-dir", type=Path, default=Path("results/audio"))
    parser.add_argument("--output", type=Path, default=Path("results/eval.parquet"))
    parser.add_argument("--summary", type=Path, default=Path("results/comparison.md"))
    parser.add_argument("--methods", nargs="+", default=["base", "dpo", "kto"])
    parser.add_argument("--reference-audio", type=Path,
                        default=Path("data/generated/reference_speaker.wav"))
    parser.add_argument("--base-model", default="OuteAI/Llama-OuteTTS-1.0-1B")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="sampling temp for eval generation; lower=conservative")
    parser.add_argument("--wer-failure-threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on eval prompts (for fast smoke test of this script)")
    args = parser.parse_args()

    set_seed(args.seed)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ----- Load eval prompts -----
    df = pd.read_parquet(args.dataset)
    eval_df = df[df["split"] == "eval"][["prompt_id", "prompt"]].drop_duplicates("prompt_id")
    eval_prompts = list(zip(eval_df["prompt_id"], eval_df["prompt"]))
    if args.limit is not None:
        eval_prompts = eval_prompts[: args.limit]
    logger.info("eval prompts: %d", len(eval_prompts))

    # ----- Reference audio for speaker_sim -----
    ref_audio: np.ndarray | None = None
    ref_sr: int | None = None
    if args.reference_audio.exists():
        ref_audio, ref_sr = load_audio(args.reference_audio)
        logger.info("reference audio: %s (sr=%d, len=%.2fs)",
                    args.reference_audio, ref_sr, len(ref_audio) / ref_sr)
    else:
        logger.warning("no reference audio at %s; speaker_sim will be skipped", args.reference_audio)

    reward_cfg = RewardConfig(weights=CompositeWeights(
        wer=0.6, utmos=0.4, speaker_sim=0.0 if ref_audio is None else 0.3,
    ))

    # ----- Prepare model paths (merge LoRA adapters as needed) -----
    model_paths: dict[str, str | None] = {}
    for method in args.methods:
        if method == "base":
            model_paths[method] = None
        elif method in ("dpo", "kto"):
            adapter_dir = find_latest_checkpoint(Path("runs") / method)
            merged_dir = adapter_dir.parent / f"{adapter_dir.name}-merged"
            merge_lora_to_dir(adapter_dir, args.base_model, merged_dir)
            model_paths[method] = str(merged_dir)
        else:
            raise ValueError(f"unknown method: {method}")

    # ----- Evaluate -----
    all_results: list[dict] = []
    for method in args.methods:
        rs = evaluate_method(
            method_name=method,
            model_path_override=model_paths[method],
            eval_prompts=eval_prompts,
            ref_audio=ref_audio,
            ref_sr=ref_sr,
            audio_dir=args.audio_dir,
            reward_cfg=reward_cfg,
            temperature=args.temperature,
        )
        all_results.extend(rs)
        # Flush partial results after each method so we don't lose all progress on a crash
        pd.DataFrame(all_results).to_parquet(args.output, index=False)

    # ----- Summarize -----
    results_df = pd.DataFrame(all_results)
    summary_df = summarize(results_df, args.methods, wer_failure_threshold=args.wer_failure_threshold)

    print("\n=== Comparison ===")
    print(summary_df.to_string(index=False))

    write_markdown_summary(summary_df, len(eval_prompts), args.summary)
    logger.info("wrote markdown summary -> %s", args.summary)
    logger.info("wrote per-sample parquet -> %s", args.output)
    logger.info("audio files under -> %s", args.audio_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
