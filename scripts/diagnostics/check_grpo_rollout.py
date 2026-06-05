"""Diagnose why GRPO rollouts score ~0.57 composite when the base model gets
~0.87 at eval time. Three hypotheses to disambiguate:

    A. The audio is genuinely bad (TRL's generation pathway differs from
       outetts.Interface.generate's).
    B. The audio is fine but our reward function path is broken.
    C. The audio has trailing garbage past <|audio_end|> that pollutes scoring.

This script reproduces one GRPO rollout outside TRL, saves the resulting WAV,
saves the iface.generate version of the same prompt as a known-good control,
saves a truncated-at-<|audio_end|> version, and scores all three with the same
reward pipeline GRPO uses. Then we ear-check the WAVs and read the scores.

Usage:
    ./run.sh grpo-rollout-check
    # then on the Mac:
    scp runpod:/workspace/tts-rl/data/generated/grpo_rollout_test.wav ~/Desktop/
    scp runpod:/workspace/tts-rl/data/generated/grpo_rollout_test_truncated.wav ~/Desktop/
    scp runpod:/workspace/tts-rl/data/generated/iface_generate_reference.wav ~/Desktop/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rewards import CompositeWeights, RewardConfig, score  # noqa: E402


def _patch_whisper_load_model_cache() -> None:
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


PROMPT_TEXT = "The candle flickered in the cool night air."
MODEL_ID = "OuteAI/Llama-OuteTTS-1.0-1B"
OUT_DIR = Path("data/generated")
REF_AUDIO = OUT_DIR / "reference_speaker.wav"


def _audio_to_numpy(audio) -> np.ndarray:
    if hasattr(audio, "audio"):
        audio = audio.audio
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    arr = np.asarray(audio).squeeze()
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
    return arr.astype(np.float32)


def _print_scores(label: str, scores: dict) -> None:
    spk = f"spk_sim={scores['speaker_sim']:.3f}" if scores["speaker_sim"] is not None else "spk_sim=N/A"
    print(f"  {label:>40}: WER={scores['wer']:.3f}  UTMOS={scores['utmos']:.2f}  {spk}  composite={scores['composite']:.3f}")


def main() -> int:
    import outetts
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== loading outetts.Interface ===")
    iface = outetts.Interface(config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        backend=outetts.Backend.HF,
    ))
    speaker = iface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    print("\n=== loading tokenizer + base model (GRPO setup, monkey-patches applied) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is not None:
        print(f"  clearing chat_template (was {tokenizer.chat_template[:60]!r})")
        tokenizer.chat_template = None
    for attr in ("add_bos_token", "add_eos_token"):
        if getattr(tokenizer, attr, None):
            setattr(tokenizer, attr, False)
            print(f"  set tokenizer.{attr} = False")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")

    prompt = iface.prompt_processor.get_completion_prompt(PROMPT_TEXT, speaker=speaker)
    print(f"\n=== prompt ===")
    print(f"  text:   {PROMPT_TEXT!r}")
    print(f"  length: {len(prompt)} chars")
    print(f"  last 200: {prompt[-200:]!r}")
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    print(f"  tokenized: {inputs.input_ids.shape[1]} tokens")

    print("\n=== generating with GRPO's sampling: temp=0.4 top_p=0.9 top_k=40 rep_penalty=1.1 max_new=2048 ===")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion_ids = out_ids[0, inputs.input_ids.shape[1]:].tolist()
    print(f"  generated: {len(completion_ids)} tokens")
    print(f"  first 20:  {completion_ids[:20]}")
    print(f"  last 20:   {completion_ids[-20:]}")

    completion_str = tokenizer.decode(completion_ids, skip_special_tokens=False)
    print(f"\n=== completion string ({len(completion_str)} chars) ===")
    print(f"  first 250: {completion_str[:250]!r}")
    print(f"  last 250:  {completion_str[-250:]!r}")

    for marker in ["<|audio_end|>", "<|im_end|>", "<|word_end|>", "<|word_start|>"]:
        idx = completion_str.find(marker)
        count = completion_str.count(marker)
        if idx >= 0:
            print(f"  '{marker}': first at char {idx} (count={count})")
        else:
            print(f"  '{marker}': NOT FOUND")

    # Decode the FULL completion via reward function's path
    print("\n=== A: full-completion decode (current reward function path) ===")
    full_ids = tokenizer.encode(completion_str, add_special_tokens=False)
    full_codes = iface.prompt_processor.extract_audio_from_tokens(full_ids)
    full_frames = len(full_codes[0]) if full_codes and full_codes[0] else 0
    print(f"  re-tokenized: {len(full_ids)} tokens")
    print(f"  audio frames: {full_frames} (~{full_frames / 75:.2f}s at 75fps)")

    if full_frames == 0:
        print("  EMPTY CODES — generation didn't produce parseable audio tokens.")
        return 1

    full_t = torch.tensor(full_codes, dtype=torch.long).unsqueeze(0).to(iface.audio_codec.device)
    full_audio = _audio_to_numpy(iface.audio_codec.decode(full_t))
    sr = int(iface.audio_codec.sr)
    full_wav = OUT_DIR / "grpo_rollout_test.wav"
    sf.write(str(full_wav), full_audio, sr)
    print(f"  decoded: {full_audio.size} samples @ {sr}Hz = {full_audio.size / sr:.2f}s")
    print(f"  saved -> {full_wav}")

    # Decode the TRUNCATED-at-<|audio_end|> version
    truncated_wav: Path | None = None
    truncated_audio: np.ndarray | None = None
    audio_end_idx = completion_str.find("<|audio_end|>")
    if audio_end_idx >= 0:
        print("\n=== B: truncated-at-<|audio_end|> decode ===")
        truncated_str = completion_str[:audio_end_idx]
        trunc_ids = tokenizer.encode(truncated_str, add_special_tokens=False)
        trunc_codes = iface.prompt_processor.extract_audio_from_tokens(trunc_ids)
        trunc_frames = len(trunc_codes[0]) if trunc_codes and trunc_codes[0] else 0
        print(f"  truncated to {len(truncated_str)} chars ({len(trunc_ids)} tokens)")
        print(f"  audio frames: {trunc_frames} (~{trunc_frames / 75:.2f}s)")
        if trunc_frames > 0:
            trunc_t = torch.tensor(trunc_codes, dtype=torch.long).unsqueeze(0).to(iface.audio_codec.device)
            truncated_audio = _audio_to_numpy(iface.audio_codec.decode(trunc_t))
            truncated_wav = OUT_DIR / "grpo_rollout_test_truncated.wav"
            sf.write(str(truncated_wav), truncated_audio, sr)
            print(f"  saved -> {truncated_wav}")
    else:
        print("\n=== B: SKIPPED (no <|audio_end|> in completion — model never emitted EOS) ===")

    # Known-good iface.generate
    print("\n=== C: iface.generate (known-good path, eval-time setup) ===")
    iface_out = iface.generate(config=outetts.GenerationConfig(
        text=PROMPT_TEXT,
        speaker=speaker,
        sampler_config=outetts.SamplerConfig(temperature=0.4),
    ))
    iface_wav = OUT_DIR / "iface_generate_reference.wav"
    iface_out.save(str(iface_wav))
    ia, isr = sf.read(str(iface_wav))
    if ia.ndim > 1:
        ia = ia.mean(axis=1)
    iface_audio = ia.astype(np.float32)
    print(f"  audio: {iface_audio.size} samples @ {isr}Hz = {iface_audio.size / isr:.2f}s")
    print(f"  saved -> {iface_wav}")

    # Score all three with the same reward pipeline GRPO uses
    print("\n=== scoring all three with the reward pipeline ===")
    ref_audio: np.ndarray | None = None
    ref_sr: int | None = None
    if REF_AUDIO.exists():
        ra, rs = sf.read(str(REF_AUDIO))
        if ra.ndim > 1:
            ra = ra.mean(axis=1)
        ref_audio = ra.astype(np.float32)
        ref_sr = int(rs)
    cfg = RewardConfig(weights=CompositeWeights(
        wer=0.6, utmos=0.4, speaker_sim=0.3 if ref_audio is not None else 0.0,
    ))

    full_scores = score(audio=full_audio, target_text=PROMPT_TEXT, sample_rate=sr,
                        reference_audio=ref_audio, reference_sr=ref_sr, config=cfg)
    _print_scores("A: full completion (current path)", full_scores)
    if truncated_audio is not None:
        trunc_scores = score(audio=truncated_audio, target_text=PROMPT_TEXT, sample_rate=sr,
                             reference_audio=ref_audio, reference_sr=ref_sr, config=cfg)
        _print_scores("B: truncated at <|audio_end|>", trunc_scores)
    iface_scores = score(audio=iface_audio, target_text=PROMPT_TEXT, sample_rate=isr,
                         reference_audio=ref_audio, reference_sr=ref_sr, config=cfg)
    _print_scores("C: iface.generate (known-good)", iface_scores)

    print("\n=== summary ===")
    print(f"  prompt: {PROMPT_TEXT!r}")
    print(f"  ear-check WAVs and check whether (A) matches what the audio actually says:")
    print(f"    A: {full_wav}")
    if truncated_wav is not None:
        print(f"    B: {truncated_wav}")
    print(f"    C: {iface_wav}")
    print()
    print("  expected patterns:")
    print("    - if C >> A and B fixes it -> trailing garbage hypothesis confirmed; fix is to truncate at <|audio_end|>")
    print("    - if C >> A and B == A -> bare model.generate is producing worse audio than iface.generate (different pathway issue)")
    print("    - if A ≈ C -> audio is fine, reward function bug somewhere else")
    return 0


if __name__ == "__main__":
    sys.exit(main())
