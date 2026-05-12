"""DPO training on the scored preference dataset.

Pipeline:
    1. Load data/dataset.parquet, filter to train split, build chosen/rejected pairs.
    2. For each pair, re-encode the chosen and rejected audio files into OuteTTS
       codec tokens, then format the full (prompt, chosen, rejected) triple as
       strings the tokenizer can round-trip.
    3. Apply LoRA on the underlying Llama base of OuteTTS (rank 16, attention + MLP).
    4. Hand off to TRL's DPOTrainer for the loss + optimization loop. With PEFT
       loaded on the policy, TRL uses adapter-disable mode as the reference policy
       (saves ~2 GB of VRAM compared to loading the base twice).

Smoke test (`--smoke-test`):
    5 pairs, batch size 1, 5 steps. Verifies the pipeline end-to-end without
    burning real compute.

Real run:
    Reads config/dpo.yaml for hyperparameters.

NOTE — outetts internals:
    Building the OuteTTS LM input (prompt + audio tokens) uses outetts library
    internals. The exact attribute names below (`iface.audio_codec`,
    `iface.prompt_processor`) may differ across outetts releases. If the script
    fails on an AttributeError, the fix is local — point the helpers below at the
    right object on the Interface — and we iterate.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_dpo_pairs, load_scored_dataset  # noqa: E402
from src.utils.lora import make_lora_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dpo")


def _resolve_audio_codec(iface):
    """Return the OuteTTS audio codec object (DAC encoder/decoder).

    Searches a few known attribute names so we tolerate minor outetts version drift.
    Raises AttributeError with a helpful message if none match.
    """
    for attr in ("audio_codec", "audio_tokenizer", "codec"):
        codec = getattr(iface, attr, None)
        if codec is not None:
            return codec
    raise AttributeError(
        "Couldn't find the audio codec on outetts.Interface. "
        "Inspect with: dir(iface). Expected one of: audio_codec, audio_tokenizer, codec."
    )


def _resolve_prompt_processor(iface):
    """Return the OuteTTS prompt-template object."""
    for attr in ("prompt_processor", "prompt_builder", "processor"):
        pp = getattr(iface, attr, None)
        if pp is not None:
            return pp
    raise AttributeError(
        "Couldn't find the prompt processor on outetts.Interface. "
        "Inspect with: dir(iface). Expected one of: prompt_processor, prompt_builder, processor."
    )


def encode_audio_path_to_token_str(iface, audio_path: Path, tokenizer) -> str:
    """Load an audio file, DAC-encode it, render as a tokenizer-roundtrippable string.

    Returns a string containing the audio-token portion of the OuteTTS sequence
    (no prompt, no special header — just the codec tokens).
    """
    import soundfile as sf

    audio, sr = sf.read(str(audio_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)

    codec = _resolve_audio_codec(iface)
    # Most codec APIs accept (audio_tensor, sample_rate). If yours has a different
    # signature, adjust here.
    encoded = codec.encode(audio_t, sample_rate=sr) if "sample_rate" in codec.encode.__code__.co_varnames \
        else codec.encode(audio_t)
    # `encoded` is typically a tensor of int token IDs already in the LM's vocab,
    # shape (B, T) or (B, n_codebooks, T). Flatten to a 1-D sequence of LM token IDs.
    if isinstance(encoded, (list, tuple)):
        encoded = encoded[0]
    if encoded.dim() == 3:
        # Interleave codebooks: cb1[0], cb2[0], cb1[1], cb2[1], ...
        encoded = encoded.permute(0, 2, 1).reshape(encoded.shape[0], -1)
    token_ids = encoded.squeeze(0).tolist()

    return tokenizer.decode(token_ids, skip_special_tokens=False)


def build_outetts_prompt_str(iface, text: str, speaker, tokenizer) -> str:
    """Build the OuteTTS LM prompt string for `text` (everything before the audio tokens).

    Falls back to a manual template if the library helper isn't reachable.
    """
    try:
        pp = _resolve_prompt_processor(iface)
        for fn_name in ("build_prompt", "get_prompt", "format_prompt"):
            fn = getattr(pp, fn_name, None)
            if fn is not None:
                return fn(text=text, speaker=speaker)
    except AttributeError as e:
        logger.warning("falling back to manual prompt template: %s", e)
    # Manual fallback — minimum-viable OuteTTS prompt, may be sub-optimal.
    return f"<|im_start|>user\n<|text_start|>{text}<|text_end|><|im_end|>\n<|im_start|>assistant\n"


def build_training_examples(iface, pairs_df, speaker, tokenizer) -> list[dict]:
    """Convert pair rows into the {prompt, chosen, rejected} text triples DPOTrainer expects."""
    examples = []
    for _, row in pairs_df.iterrows():
        prompt_str = build_outetts_prompt_str(iface, row["prompt"], speaker, tokenizer)
        chosen_str = encode_audio_path_to_token_str(iface, Path(row["chosen_audio_path"]), tokenizer)
        rejected_str = encode_audio_path_to_token_str(iface, Path(row["rejected_audio_path"]), tokenizer)
        examples.append({"prompt": prompt_str, "chosen": chosen_str, "rejected": rejected_str})
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/dpo.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="cap on number of training pairs; defaults to all")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["seed"])

    # ----- Load dataset and build preference pairs -----
    df = load_scored_dataset(Path(cfg["dataset_path"]))
    pairs = build_dpo_pairs(df, split="train")
    logger.info("loaded %d preference pairs (train split)", len(pairs))
    if args.smoke_test:
        pairs = pairs.head(5)
        logger.info("--smoke-test: capping to 5 pairs")
    elif args.max_pairs is not None:
        pairs = pairs.head(args.max_pairs)

    # Margin diagnostics so we know what kind of signal DPO is being asked to learn
    if len(pairs) > 0:
        m = pairs["margin"]
        logger.info("composite margin: min=%.3f mean=%.3f max=%.3f", m.min(), m.mean(), m.max())

    # ----- Load OuteTTS interface (for prompt processor + audio codec) -----
    logger.info("loading OuteTTS interface (HF backend)")
    import outetts

    iface = outetts.Interface(
        config=outetts.ModelConfig.auto_config(
            model=outetts.Models.VERSION_1_0_SIZE_1B,
            backend=outetts.Backend.HF,
        )
    )
    speaker = iface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    # ----- Tokenizer + base model for training -----
    # We load the LM weights afresh via transformers (rather than reusing iface.model)
    # so the training graph has full control over precision, LoRA insertion, etc.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading base Llama weights: %s", cfg["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # ----- Build training dataset -----
    logger.info("encoding %d audio pairs to LM-token strings", len(pairs))
    t0 = time.time()
    examples = build_training_examples(iface, pairs, speaker, tokenizer)
    logger.info("  encoded in %.1fs", time.time() - t0)
    train_ds = Dataset.from_list(examples)
    logger.info("training dataset: %d examples", len(train_ds))

    # Free the outetts inference interface — we don't need it during training, and
    # holding it costs ~2 GB of VRAM.
    del iface
    torch.cuda.empty_cache()

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

    # ----- DPOTrainer -----
    from trl import DPOConfig, DPOTrainer

    run_name = f"dpo-{'smoke-' if args.smoke_test else ''}{int(time.time())}"
    dpo_args = DPOConfig(
        output_dir=str(Path(cfg["output_dir"]) / run_name),
        beta=cfg["beta"],
        max_steps=5 if args.smoke_test else cfg["max_steps"],
        per_device_train_batch_size=1 if args.smoke_test else cfg["batch_size"],
        gradient_accumulation_steps=1 if args.smoke_test else cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"] if not args.smoke_test else 999_999,
        bf16=True,
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        report_to=[] if args.smoke_test else ["wandb"],
        run_name=run_name,
        seed=cfg["seed"],
        remove_unused_columns=False,
    )

    # With PEFT loaded on the policy, passing ref_model=None tells DPOTrainer to
    # disable the adapter to obtain reference logprobs. Saves a full model copy.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    logger.info("starting DPO training: run=%s steps=%d", run_name, dpo_args.max_steps)
    trainer.train()
    trainer.save_model()
    logger.info("training complete. checkpoints under %s", dpo_args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
