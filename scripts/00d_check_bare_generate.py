"""Compare iface.generate() (known working) to bare model.generate() on the same
prompt + same sampling, to isolate where the GRPO 'model emits text instead of
audio tokens' bug lives.

If both produce audio tokens -> TRL is the culprit (chat-template, dataset
preprocessing, etc.).
If iface.generate works but bare model.generate doesn't -> outetts.Interface
does something we need to replicate when invoking generate from TRL.
If neither works -> the prompt construction itself is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_completion(token_ids: list[int], iface) -> tuple[int, str]:
    """Return (n_audio_code_frames, first_300_chars_decoded)."""
    codes = iface.prompt_processor.extract_audio_from_tokens(token_ids)
    n_frames = len(codes[0]) if codes and codes[0] else 0
    text = iface.tokenizer.decode(token_ids[:200], skip_special_tokens=False) if hasattr(iface, 'tokenizer') else "[no tokenizer on iface]"
    return n_frames, text[:300]


def main() -> int:
    import outetts
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_ID = "OuteAI/Llama-OuteTTS-1.0-1B"
    PROMPT_TEXT = "The candle flickered in the cool night air."

    print(f"\n=== loading outetts.Interface ===")
    iface = outetts.Interface(config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        backend=outetts.Backend.HF,
    ))
    speaker = iface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    # ----- 1. iface.generate -----
    print(f"\n=== [A] iface.generate (known working path) ===")
    out = iface.generate(config=outetts.GenerationConfig(
        text=PROMPT_TEXT,
        speaker=speaker,
        sampler_config=outetts.SamplerConfig(temperature=0.4),
    ))
    # out has .audio (the waveform tensor). For our purposes, just confirm it produced something audible.
    audio_attr = getattr(out, 'audio', None)
    if audio_attr is not None:
        if hasattr(audio_attr, 'shape'):
            print(f"  [A] iface.generate -> audio shape {tuple(audio_attr.shape)} (success)")
        else:
            print(f"  [A] iface.generate -> audio type {type(audio_attr).__name__}")
    else:
        print(f"  [A] iface.generate -> {type(out).__name__}: {repr(out)[:200]}")

    # ----- 2. bare model.generate -----
    print(f"\n=== [B] bare model.generate (TRL-style invocation) ===")
    print(f"  loading transformers AutoModelForCausalLM...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"  tokenizer.chat_template is None: {tokenizer.chat_template is None}")
    if tokenizer.chat_template is not None:
        print(f"  tokenizer.chat_template (first 200): {tokenizer.chat_template[:200]!r}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")

    # Build prompt the SAME way GRPO script does
    prompt = iface.prompt_processor.get_completion_prompt(PROMPT_TEXT, speaker=speaker)
    print(f"  prompt length: {len(prompt)} chars")
    print(f"  prompt last 200: {prompt[-200:]!r}")

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    print(f"  tokenized: {inputs.input_ids.shape[1]} tokens")
    print(f"  last 20 input token IDs: {inputs.input_ids[0, -20:].tolist()}")

    print(f"\n  generating with OuteTTS canonical sampling (temp=0.4, top_p=0.9, top_k=40, rep_penalty=1.1)...")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )
    completion_ids = out_ids[0, inputs.input_ids.shape[1]:].tolist()
    print(f"  generated {len(completion_ids)} tokens")
    print(f"  first 20 completion token IDs: {completion_ids[:20]}")
    completion_str = tokenizer.decode(completion_ids, skip_special_tokens=False)
    print(f"  completion (first 400 chars): {completion_str[:400]!r}")

    n_frames = 0
    try:
        codes = iface.prompt_processor.extract_audio_from_tokens(completion_ids)
        n_frames = len(codes[0]) if codes and codes[0] else 0
    except Exception as e:
        print(f"  extract_audio_from_tokens raised: {e}")
    print(f"  -> {n_frames} audio code frames extracted from completion")

    if n_frames > 0:
        print(f"\nPASS — bare model.generate produces audio tokens. The GRPO bug is in TRL's prompt handling.")
        return 0
    else:
        print(f"\nFAIL — bare model.generate also produces text. The problem is below TRL (prompt format, generation invocation, or model behavior).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
