"""KTO training on the scored preference dataset.

Same dataset as DPO (data/dataset.parquet), same OuteTTS encoding pipeline,
same LoRA setup — only the loss differs. KTO works on per-candidate binary
labels (desirable vs undesirable) instead of paired (chosen, rejected).

Label assignment:
    - Sort the train-split candidates by composite score.
    - Top `desirable_quantile` fraction (e.g. top 33%) → desirable.
    - Bottom `1 - undesirable_quantile` fraction (e.g. bottom 33%) → undesirable.
    - Middle is dropped.
    The thresholds come from the dataset's own distribution, so they adapt
    to whatever this particular Phase 2a run scored.

Smoke test (`--smoke-test`):
    10 labeled examples, batch size 2, 5 steps. Verifies the pipeline end-to-end.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_kto_labels, load_scored_dataset  # noqa: E402
from src.utils.lora import make_lora_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("kto")


def _patch_whisper_load_model_cache() -> None:
    """Same monkey-patch as the DPO script: outetts reloads Whisper from
    disk on every create_speaker call (~45s). Cache the first load.
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


def _patch_kto_trainer_sampler() -> None:
    """TRL 0.16/0.17's KTOTrainer._get_train_sampler takes only `self`, but
    transformers >=4.50 calls it with the dataset as an extra positional arg.
    Swallow the extra arg so the existing implementation still works.
    """
    from trl import KTOTrainer

    _orig = KTOTrainer._get_train_sampler

    def _wrapped(self, *args, **kwargs):
        return _orig(self)

    KTOTrainer._get_train_sampler = _wrapped


_patch_kto_trainer_sampler()


def build_outetts_prompt_str(iface, prompt_text: str) -> str:
    return iface.prompt_processor.get_completion_prompt(prompt_text)


def encode_audio_to_completion(iface, audio_path: Path, prompt_text: str) -> str:
    """Same canonical OuteTTS encoding pipeline used in DPO: Whisper align +
    DAC encode + features → get_training_prompt → strip the inference-prompt
    prefix so what's left is the completion the model would generate.
    """
    spk = iface.create_speaker(audio_path=str(audio_path), transcript=prompt_text)
    spk["text"] = prompt_text
    full = iface.prompt_processor.get_training_prompt(spk)
    prefix = iface.prompt_processor.get_completion_prompt(prompt_text)
    if not full.startswith(prefix):
        raise ValueError(
            f"get_training_prompt did not start with get_completion_prompt for {audio_path}."
        )
    return full[len(prefix):]


def build_training_examples(iface, labeled_df) -> list[dict]:
    """Convert labeled rows into the {prompt, completion, label} triples KTOTrainer expects."""
    examples = []
    for _, row in labeled_df.iterrows():
        prompt_text = row["prompt"]
        prompt_str = build_outetts_prompt_str(iface, prompt_text)
        completion_str = encode_audio_to_completion(iface, Path(row["audio_path"]), prompt_text)
        examples.append(
            {
                "prompt": prompt_str,
                "completion": completion_str,
                "label": bool(row["desirable"]),
            }
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/kto.yaml"))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(cfg["seed"])

    # ----- Load dataset and derive per-candidate labels -----
    df = load_scored_dataset(Path(cfg["dataset_path"]))
    train_df = df[df["split"] == "train"]
    if len(train_df) == 0:
        raise RuntimeError("No train-split rows in dataset.")

    # Quantile-based thresholds so we adapt to whatever distribution scoring produced.
    desirable_threshold = float(train_df["composite"].quantile(cfg["desirable_quantile"]))
    undesirable_threshold = float(train_df["composite"].quantile(cfg["undesirable_quantile"]))
    logger.info(
        "thresholds: desirable >%.3f (quantile %.2f), undesirable <%.3f (quantile %.2f)",
        desirable_threshold, cfg["desirable_quantile"],
        undesirable_threshold, cfg["undesirable_quantile"],
    )

    labeled = build_kto_labels(
        df,
        split="train",
        desirable_threshold=desirable_threshold,
        undesirable_threshold=undesirable_threshold,
    )
    n_des = int(labeled["desirable"].sum())
    n_und = int((~labeled["desirable"]).sum())
    logger.info("labeled candidates: %d desirable, %d undesirable, %d dropped (middle)",
                n_des, n_und, len(train_df) - len(labeled))

    if args.smoke_test:
        # 10 examples, half from each side so the smoke test exercises both branches
        des_head = labeled[labeled["desirable"]].head(5)
        und_head = labeled[~labeled["desirable"]].head(5)
        labeled = pd.concat([des_head, und_head]).reset_index(drop=True)
        logger.info("--smoke-test: %d examples (%d desirable, %d undesirable)",
                    len(labeled), len(des_head), len(und_head))
    elif args.max_examples is not None:
        labeled = labeled.head(args.max_examples)

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

    # ----- Encode audio → token sequences -----
    logger.info("encoding %d labeled candidates via Whisper alignment + DAC", len(labeled))
    t0 = time.time()
    examples = build_training_examples(iface, labeled)
    logger.info("  encoded in %.1fs (~%.1fs per audio)", time.time() - t0, (time.time() - t0) / max(1, len(labeled)))
    train_ds = Dataset.from_list(examples)

    # Free outetts to reclaim VRAM before training
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

    # ----- KTOTrainer -----
    from trl import KTOConfig, KTOTrainer

    run_name = f"kto-{'smoke-' if args.smoke_test else ''}{int(time.time())}"
    kto_args = KTOConfig(
        output_dir=str(Path(cfg["output_dir"]) / run_name),
        beta=cfg["beta"],
        desirable_weight=cfg["desirable_weight"],
        undesirable_weight=cfg["undesirable_weight"],
        max_steps=5 if args.smoke_test else cfg["max_steps"],
        per_device_train_batch_size=2 if args.smoke_test else cfg["batch_size"],
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

    # PEFT-disable-as-reference, same pattern as DPO.
    trainer = KTOTrainer(
        model=model,
        ref_model=None,
        args=kto_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    logger.info("starting KTO training: run=%s steps=%d", run_name, kto_args.max_steps)
    trainer.train()
    trainer.save_model()
    logger.info("training complete. checkpoints under %s", kto_args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
