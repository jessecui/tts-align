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


def _patch_whisper_load_model_cache() -> None:
    """outetts's create_speaker re-loads the Whisper checkpoint from disk on
    every call (~45s each on this hardware). It's the dominant cost when
    encoding the DPO dataset. Monkey-patch whisper.load_model so subsequent
    calls return the already-loaded model.
    """
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


def build_outetts_prompt_str(iface, prompt_text: str) -> str:
    """The DPO 'prompt': the OuteTTS inference prompt for `prompt_text`.

    `prompt_processor.get_completion_prompt(text)` returns the canonical
    `<|im_start|>\\n<|text_start|>{text}<|text_end|>\\n<|audio_start|>\\n` —
    exactly what the model sees at inference time before it starts emitting
    audio tokens. We use this for both chosen and rejected so they share an
    identical conditioning prefix (a requirement for DPO).
    """
    return iface.prompt_processor.get_completion_prompt(prompt_text)


def encode_audio_to_dpo_completion(iface, audio_path: Path, prompt_text: str) -> str:
    """The DPO 'chosen' or 'rejected' completion string for one audio file.

    OuteTTS's canonical training format interleaves text and audio codes
    word-by-word, with prosodic features. To match that distribution we ingest
    the audio through `iface.create_speaker` (Whisper alignment + DAC encode +
    feature extraction), then render with `get_training_prompt`.

    We strip the inference-prompt prefix so what's returned is only the
    'completion' tokens — the audio-rendering portion the model would generate.
    We also override `spk['text']` to our supplied `prompt_text` so both chosen
    and rejected produce completions that share an identical prompt header
    (Whisper occasionally mis-transcribes bad TTS samples, which would
    otherwise corrupt the prefix).
    """
    spk = iface.create_speaker(audio_path=str(audio_path), transcript=prompt_text)
    spk["text"] = prompt_text
    full = iface.prompt_processor.get_training_prompt(spk)
    prefix = iface.prompt_processor.get_completion_prompt(prompt_text)
    if not full.startswith(prefix):
        raise ValueError(
            f"get_training_prompt did not start with get_completion_prompt for {audio_path}. "
            f"Prefix bytes don't match — check outetts version compatibility."
        )
    return full[len(prefix):]


def build_training_examples(iface, pairs_df) -> list[dict]:
    """Convert pair rows into the {prompt, chosen, rejected} text triples DPOTrainer expects."""
    examples = []
    for _, row in pairs_df.iterrows():
        prompt_text = row["prompt"]
        prompt_str = build_outetts_prompt_str(iface, prompt_text)
        chosen_str = encode_audio_to_dpo_completion(iface, Path(row["chosen_audio_path"]), prompt_text)
        rejected_str = encode_audio_to_dpo_completion(iface, Path(row["rejected_audio_path"]), prompt_text)
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
    logger.info("encoding %d audio pairs to LM-token strings via Whisper alignment + DAC", len(pairs))
    t0 = time.time()
    examples = build_training_examples(iface, pairs)
    logger.info("  encoded in %.1fs (~%.1fs per audio)", time.time() - t0, (time.time() - t0) / max(1, 2 * len(pairs)))
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
    # TRL >=0.16 renamed `tokenizer` -> `processing_class`. We unify here.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    logger.info("starting DPO training: run=%s steps=%d", run_name, dpo_args.max_steps)
    trainer.train()
    trainer.save_model()
    logger.info("training complete. checkpoints under %s", dpo_args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
