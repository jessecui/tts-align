"""GRPO training on the same prompts as DPO/KTO, but online.

Pipeline:
    1. Build a dataset of *prompts only* (no completions; rollouts come from the policy).
    2. Wrap our reward pipeline as a callable that:
        completion_str -> token IDs -> extract audio codes -> DAC decode -> waveform
            -> WER + UTMOS + speaker_sim -> composite reward.
    3. Hand off to trl.GRPOTrainer with vLLM-accelerated rollouts.

NOTE: this assumes the audio round-trip works as the diagnostic checks.
Run `./run.sh roundtrip-check` first.

Smoke test (`--smoke-test`):
    5 prompts, 5 steps, num_generations=2 (smaller group), vLLM disabled.
    Verifies the pipeline end-to-end without burning real compute.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_scored_dataset  # noqa: E402
from src.rewards import CompositeWeights, RewardConfig, score  # noqa: E402
from src.utils.lora import make_lora_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("grpo")


def _patch_whisper_load_model_cache() -> None:
    """Same monkey-patch as DPO/KTO scripts: cache Whisper model loads."""
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


# Regex to pull the target text back out of an OuteTTS inference prompt.
# get_completion_prompt(text) produces "<|im_start|>\n<|text_start|>{text}<|text_end|>\n<|audio_start|>\n".
_TEXT_FROM_PROMPT_RE = re.compile(r"<\|text_start\|>(.*?)<\|text_end\|>", re.DOTALL)


def extract_text_from_prompt(prompt_str: str) -> str:
    m = _TEXT_FROM_PROMPT_RE.search(prompt_str)
    if not m:
        raise ValueError(f"couldn't find <|text_start|>...<|text_end|> in prompt: {prompt_str[:200]}")
    return m.group(1).strip()


def make_reward_fn(iface, tokenizer, ref_audio: np.ndarray | None, ref_sr: int | None, reward_cfg: RewardConfig):
    """Build a TRL-compatible reward callable.

    TRL signature: `reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]`.
    `prompts` are the dataset prompts; `completions` are what the policy generated.
    Both are decoded text strings, not token IDs.

    For each completion we:
        1. Tokenize back to integer IDs.
        2. Use OuteTTS to extract audio codes from the token stream.
        3. DAC-decode codes -> waveform.
        4. Run our composite reward.
    Catastrophic failures (un-decodable sequences, empty audio, etc.) score 0.
    """
    target_sr = int(getattr(iface.audio_codec, "sr", 24000))

    def _decode_to_audio(completion_str: str) -> tuple[np.ndarray, int] | None:
        ids = tokenizer.encode(completion_str, add_special_tokens=False)
        try:
            codes = iface.prompt_processor.extract_audio_from_tokens(ids)
        except Exception as e:
            logger.warning("extract_audio_from_tokens failed: %s", e)
            return None
        # `codes` comes back as list[list[int]] (one inner list per DAC codebook).
        # audio_codec.decode wants an int64 tensor of shape (B=1, n_codebooks=2, T)
        # on the codec's device. Same conversion we use in the round-trip diagnostic.
        if not codes or not codes[0]:
            return None
        try:
            codes_t = torch.tensor(codes, dtype=torch.long).unsqueeze(0).to(iface.audio_codec.device)
        except Exception as e:
            logger.warning("codes tensor build failed: %s", e)
            return None
        try:
            audio = iface.audio_codec.decode(codes_t)
        except Exception as e:
            logger.warning("DAC decode failed: %s", e)
            return None
        # Normalize to (T,) float32 numpy on CPU
        if hasattr(audio, "audio"):
            audio = audio.audio
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        arr = np.asarray(audio).squeeze()
        if arr.ndim > 1:
            arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
        arr = arr.astype(np.float32)
        if arr.size < target_sr * 0.2:   # less than 0.2s of audio -> garbage
            return None
        return arr, target_sr

    # Diagnostic counter — every Nth reward batch we log details of the first
    # completion so we can see whether the model is producing parseable audio tokens.
    call_count = {"n": 0}

    def reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        call_count["n"] += 1
        verbose = call_count["n"] <= 3 or call_count["n"] % 25 == 0

        # TRL decodes completions with skip_special_tokens=True, which strips
        # every <|word_start|>, <|c1_N|>, <|c2_N|> — exactly the tokens we need
        # to recover audio codes. We have to use the raw token IDs from kwargs.
        # Different TRL versions name this kwarg differently; check several.
        if verbose and call_count["n"] == 1:
            logger.info("[reward debug] reward_func kwargs keys: %s", list(kwargs.keys()))
            for k, v in kwargs.items():
                if hasattr(v, '__len__'):
                    sample = v[0] if len(v) > 0 else None
                    sample_repr = type(sample).__name__
                    if hasattr(sample, '__len__'):
                        sample_repr += f"(len={len(sample)})"
                    elif sample is not None:
                        sample_repr += f"={sample!r}"[:60]
                    logger.info("[reward debug]   kwargs[%r]: type=%s len=%d sample=%s",
                                k, type(v).__name__, len(v), sample_repr)

        completion_ids_list = (
            kwargs.get("completion_ids")
            or kwargs.get("completions_ids")
            or kwargs.get("response_ids")
            or kwargs.get("responses")
        )
        if verbose and completion_ids_list is None:
            logger.warning("[reward debug] no completion_ids found in kwargs; cannot recover audio tokens")

        rewards: list[float] = []
        n_empty_codes = n_short_audio = n_decode_err = n_score_err = n_ok = 0
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            try:
                target_text = extract_text_from_prompt(prompt)
            except Exception as e:
                logger.warning("bad prompt: %s", e)
                rewards.append(0.0)
                continue

            # ---- decode (inline so we can introspect each step) ----
            if completion_ids_list is not None and i < len(completion_ids_list):
                ids = list(completion_ids_list[i]) if not isinstance(completion_ids_list[i], list) else completion_ids_list[i]
            else:
                ids = tokenizer.encode(completion, add_special_tokens=False)
            try:
                codes = iface.prompt_processor.extract_audio_from_tokens(ids)
            except Exception as e:
                if verbose and i == 0:
                    logger.warning("[reward debug] extract_audio_from_tokens raised: %s", e)
                n_decode_err += 1
                rewards.append(0.0)
                continue

            n_code_frames = len(codes[0]) if codes and codes[0] else 0
            if n_code_frames == 0:
                if verbose and i == 0:
                    snippet = completion[:300].replace("\n", " ")
                    n_ids = len(ids)
                    logger.warning("[reward debug] empty codes. ids_len=%d first_20_ids=%s completion_decoded[:300]=%r",
                                   n_ids, ids[:20], snippet)
                n_empty_codes += 1
                rewards.append(0.0)
                continue

            try:
                codes_t = torch.tensor(codes, dtype=torch.long).unsqueeze(0).to(iface.audio_codec.device)
                audio = iface.audio_codec.decode(codes_t)
            except Exception as e:
                if verbose and i == 0:
                    logger.warning("[reward debug] decode raised: %s", e)
                n_decode_err += 1
                rewards.append(0.0)
                continue

            if hasattr(audio, "audio"):
                audio = audio.audio
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            arr = np.asarray(audio).squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
            arr = arr.astype(np.float32)
            if arr.size < target_sr * 0.2:
                n_short_audio += 1
                rewards.append(0.0)
                continue

            try:
                scores = score(
                    audio=arr,
                    target_text=target_text,
                    sample_rate=target_sr,
                    reference_audio=ref_audio,
                    reference_sr=ref_sr,
                    config=reward_cfg,
                )
                rewards.append(float(scores["composite"]))
                n_ok += 1
                if verbose and i == 0:
                    logger.info("[reward debug] OK. n_code_frames=%d, audio_len=%d (%.2fs), wer=%.3f utmos=%.2f composite=%.3f",
                                n_code_frames, arr.size, arr.size / target_sr,
                                scores["wer"], scores["utmos"], scores["composite"])
            except Exception as e:
                if verbose and i == 0:
                    logger.warning("[reward debug] scoring raised: %s", e)
                n_score_err += 1
                rewards.append(0.0)

        if verbose:
            logger.info("[reward batch %d] ok=%d empty_codes=%d short_audio=%d decode_err=%d score_err=%d | total=%d mean=%.3f",
                        call_count["n"], n_ok, n_empty_codes, n_short_audio, n_decode_err, n_score_err,
                        len(rewards), sum(rewards) / max(1, len(rewards)))
        return rewards

    return reward_func


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/grpo.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="cap on training prompts (default: use all train split)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["seed"])

    # ----- Prompts (train split, deduped) -----
    df = load_scored_dataset(Path(cfg["dataset_path"]))
    train_df = df[df["split"] == "train"][["prompt_id", "prompt"]].drop_duplicates("prompt_id")
    logger.info("train prompts available: %d", len(train_df))

    # ----- OuteTTS interface (needed for prompt building + audio decoding inside the reward fn) -----
    logger.info("loading OuteTTS interface (HF backend)")
    import outetts

    iface = outetts.Interface(config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        backend=outetts.Backend.HF,
    ))

    # ----- Build prompt strings (canonical inference prompt for each text) -----
    # We MUST pass a speaker reference here. OuteTTS-1.0 was trained with a
    # speaker reference in every generation prompt; without one, the bare
    # `<|text_start|>X<|text_end|>\n<|audio_start|>\n` prompt drives the model
    # to emit plain English text (echoing the input) instead of codec tokens.
    # `EN-FEMALE-1-NEUTRAL` is the same default speaker we used in Phases 2/5.
    speaker = iface.load_default_speaker("EN-FEMALE-1-NEUTRAL")
    logger.info("using default speaker EN-FEMALE-1-NEUTRAL for all generation prompts")

    prompts: list[dict] = []
    for _, row in train_df.iterrows():
        prompt_str = iface.prompt_processor.get_completion_prompt(row["prompt"], speaker=speaker)
        prompts.append({"prompt": prompt_str})

    if args.smoke_test:
        prompts = prompts[:5]
        logger.info("--smoke-test: 5 prompts")
    elif args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    train_ds = Dataset.from_list(prompts)
    logger.info("training dataset: %d prompts", len(train_ds))

    # ----- Tokenizer + base model -----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Null out the chat_template so TRL doesn't wrap our already-formatted
    # OuteTTS prompts. Our prompts are pre-built by outetts.prompt_processor
    # and need to reach the model verbatim.
    if tokenizer.chat_template is not None:
        logger.info("clearing tokenizer.chat_template (was %r) to prevent TRL from re-wrapping prompts",
                    tokenizer.chat_template[:60])
        tokenizer.chat_template = None

    # Also disable BOS / EOS auto-insertion (matches the standalone diagnostic
    # which used add_special_tokens=False).
    for attr in ("add_bos_token", "add_eos_token"):
        if getattr(tokenizer, attr, None):
            logger.info("setting tokenizer.%s = False", attr)
            setattr(tokenizer, attr, False)

    # Force tokenizer.batch_decode to preserve special tokens. TRL's GRPO
    # version on this box passes no kwargs to reward_funcs (only prompts +
    # completions), so the raw generated token IDs aren't directly available.
    # By making batch_decode preserve special tokens, the `completion` strings
    # TRL hands us still contain <|word_start|>, <|c1_N|>, <|c2_N|>, etc., which
    # we can re-encode in the reward function to recover the original IDs.
    _orig_batch_decode = tokenizer.batch_decode

    def _batch_decode_no_skip(*args, **kwargs):
        kwargs["skip_special_tokens"] = False
        return _orig_batch_decode(*args, **kwargs)

    tokenizer.batch_decode = _batch_decode_no_skip
    logger.info("monkey-patched tokenizer.batch_decode to preserve special tokens "
                "(needed to recover audio-token IDs in the reward function)")

    logger.info("loading base Llama weights: %s", cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # ----- Reference audio for speaker_sim -----
    ref_audio: np.ndarray | None = None
    ref_sr: int | None = None
    ref_path = Path(cfg["reference_audio_path"])
    if ref_path.exists():
        a, s = sf.read(str(ref_path), always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)
        ref_audio = a.astype(np.float32)
        ref_sr = int(s)
    else:
        logger.warning("no reference audio at %s; speaker_sim will be 0-weighted", ref_path)

    reward_cfg = RewardConfig(weights=CompositeWeights(
        wer=cfg["reward_weights"]["wer"],
        utmos=cfg["reward_weights"]["utmos"],
        speaker_sim=cfg["reward_weights"]["speaker_sim"] if ref_audio is not None else 0.0,
    ))

    reward_func = make_reward_fn(iface, tokenizer, ref_audio, ref_sr, reward_cfg)

    # ----- LoRA -----
    from peft import get_peft_model

    lora_cfg = make_lora_config(
        r=cfg["lora"]["r"],
        alpha=cfg["lora"]["alpha"],
        dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ----- GRPOTrainer -----
    from trl import GRPOConfig, GRPOTrainer

    run_name = f"grpo-{'smoke-' if args.smoke_test else ''}{int(time.time())}"
    grpo_args = GRPOConfig(
        output_dir=str(Path(cfg["output_dir"]) / run_name),
        beta=cfg["beta"],
        num_generations=2 if args.smoke_test else cfg["num_generations"],
        num_iterations=cfg["num_iterations"],
        max_steps=5 if args.smoke_test else cfg["max_steps"],
        # GRPO requires effective batch >= num_generations AND evenly divisible by it.
        # Smoke test: batch=2, accum=1, num_gen=2 -> effective 2, valid.
        per_device_train_batch_size=2 if args.smoke_test else cfg["batch_size"],
        gradient_accumulation_steps=1 if args.smoke_test else cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"] if not args.smoke_test else 999_999,
        bf16=True,
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg.get("top_k", 40),
        repetition_penalty=cfg.get("repetition_penalty", 1.1),
        use_vllm=False if args.smoke_test else cfg["use_vllm"],
        vllm_gpu_memory_utilization=cfg["vllm_gpu_memory_utilization"],
        report_to=[] if args.smoke_test else ["wandb"],
        run_name=run_name,
        seed=cfg["seed"],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    logger.info("starting GRPO training: run=%s steps=%d", run_name, grpo_args.max_steps)
    trainer.train()
    trainer.save_model()
    logger.info("training complete. checkpoints under %s", grpo_args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
