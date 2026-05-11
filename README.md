# tts-rl: DPO, KTO, and GRPO on a small autoregressive TTS model

Exploratory hobby project demonstrating three preference-optimization methods — **DPO**, **KTO**, and **GRPO** —
applied to the same small autoregressive codec-LM TTS model, with comparable automatic evals.

**Status:** Phase 1 (reward pipeline + repo scaffold). Training scripts are not yet written.

## Plan

| Phase | Scope                                              | Status |
|-------|----------------------------------------------------|--------|
| 1     | Repo scaffold + reward pipeline + smoke test       | In progress |
| 2     | Preference dataset generation + DPO end-to-end     | Not started |
| 3     | KTO                                                | Not started |
| 4     | GRPO (online; vLLM-accelerated sampling)           | Not started |
| 5     | Held-out comparison eval + writeup                 | Not started |

## Model choice

**Base:** [`OuteAI/Llama-OuteTTS-1.0-1B`](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B)
— OuteTTS 1.0, Llama 3.2-1B fine-tuned for TTS. Uses DAC codec
([`ibm-research/DAC.speech.v1.0`](https://huggingface.co/ibm-research/DAC.speech.v1.0))
as the audio decoder.

Why this model:
- Standard `LlamaForCausalLM` underneath → TRL's `DPOTrainer`/`KTOTrainer`/`GRPOTrainer`
  work out of the box.
- Supports voice cloning via reference audio, which makes `speaker_sim` a meaningful
  reward signal.
- vLLM officially supports batched inference for this model (matters for GRPO).
- ~1B params fits comfortably on a single 24GB GPU with LoRA.

## Quickstart (from a fresh clone on a rented Linux + NVIDIA box)

Assumes a clean Ubuntu 22.04 image with CUDA 12.4 and Python 3.11.

```bash
# 1. Get the code and install Python deps (~3-5 min, ~2GB of wheels)
git clone <your-fork-url> tts-rl
cd tts-rl
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not already installed
uv sync

# 2. Set your wandb key (get it from https://wandb.ai/authorize)
export WANDB_API_KEY=<your-key>

# 3. Phase 1: verify the reward pipeline works on a few generated samples (~5-8 min)
#    Downloads OuteTTS-1.0-1B (~2GB), DAC decoder, Whisper-small (~500MB), UTMOS, ECAPA-TDNN.
uv run python scripts/00_smoke_test_rewards.py --voice-cloning

# 4. Phase 1: confirm vLLM works with the chosen model (~2-3 min)
uv add 'vllm==0.6.4.post1'
uv run python scripts/00b_check_vllm_compat.py

# --- Below are not yet implemented; placeholder commands for future phases ---
# uv run python scripts/01_generate_dataset.py        # Phase 2: ~20-30 min
# uv run python scripts/02_train_dpo.py               # Phase 2: ~20-30 min
# uv run python scripts/03_train_kto.py               # Phase 3: ~20-30 min
# uv run python scripts/04_train_grpo.py              # Phase 4: ~1-3 hours
# uv run python scripts/05_evaluate_all.py            # Phase 5: ~20 min
```

All training scripts will accept `--smoke-test` to run 5 steps with batch size 1 for a
final pipeline check before committing to a real run.

## Rented compute setup (RunPod)

I develop from a MacBook with no GPU. All training runs on a rented RunPod A100 40GB.
Local machine = code, git, wandb dashboard. Remote = training.

### One-time RunPod setup

1. Sign up at [runpod.io](https://runpod.io) and add a payment method ($10 is plenty for Phase 1).
2. Generate an SSH key pair on the Mac if you don't have one (`ssh-keygen -t ed25519`) and
   paste the public key into RunPod → Settings → SSH Public Keys.

### Launching a pod

- **GPU type:** A100 40GB PCIe (or A100 80GB if you want headroom). H100 80GB also works but
  costs ~2x for this workload.
- **Template:** `RunPod Pytorch 2.4` (or any recent `runpod/pytorch:*-cuda12.4-*` image).
- **Disk:** 50 GB container disk + a **persistent network volume** of 50 GB mounted at
  `/workspace`. The network volume survives pod stop/start; the container disk does not.
  Use it for HF model cache and checkpoints.
- **Expose ports:** SSH (22) — enable "Start SSH Daemon" in the pod config.
- **Spot vs on-demand:** spot is ~half price but can be reclaimed without warning; for
  training runs longer than 30 minutes prefer on-demand.

### Connecting via SSH (VS Code Remote)

After the pod is "Running", grab the SSH command from the pod's "Connect" tab. It looks
like `ssh root@<podid>-<rand>.proxy.runpod.net -i ~/.ssh/id_ed25519 -p <port>`.

In `~/.ssh/config` on the Mac:
```
Host runpod
    HostName <podid>-<rand>.proxy.runpod.net
    User root
    Port <port>
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 12
```

Then in VS Code: `Cmd-Shift-P` → `Remote-SSH: Connect to Host` → `runpod`.

### Persistent env on the box

```bash
# On the rented box, append to ~/.bashrc on the persistent volume so it survives pod restarts.
cat >> /workspace/.bashrc_extra <<'EOF'
export HF_HOME=/workspace/hf_cache
export WANDB_API_KEY=<your-key>
export WANDB_PROJECT=tts-rl
EOF
echo 'source /workspace/.bashrc_extra' >> ~/.bashrc
source ~/.bashrc
```

### Long runs without losing your session

Always run training under `tmux` so an SSH disconnect doesn't kill it:
```bash
tmux new -s train          # start a session
# ... run training ...
# Ctrl-b d                 # detach
tmux attach -t train       # reattach later (after reconnecting SSH)
```

### Getting results back to the Mac

```bash
# On the Mac, with the SSH config alias above:
rsync -avz --progress runpod:/workspace/tts-rl/runs/ ./runs/
rsync -avz --progress runpod:/workspace/tts-rl/results/ ./results/
```

### Stopping the pod when done

Always **stop** (not just disconnect) the pod when you're done for the day — RunPod charges
for compute while it's running, even when idle. The network volume keeps your data for the
next session. Restarting picks up exactly where you left off (cached models, checkpoints,
HF cache all persist).

## Reward pipeline

The composite reward function (`src/rewards/composite.py`) is the single entrypoint used by
both offline dataset scoring (Phase 2) and online GRPO reward computation (Phase 4).

```python
from src.rewards import score
out = score(
    audio=audio_array,           # 1-D float32 mono
    target_text="...",           # ground-truth transcript
    sample_rate=24000,           # OuteTTS native SR
    reference_audio=ref_array,   # optional; required if speaker_sim is weighted
    reference_sr=24000,
)
# {"wer": 0.07, "utmos": 4.12, "speaker_sim": 0.74, "composite": 0.81}
```

Sub-metrics:
- **WER** via Whisper-small + jiwer. Range `[0, ∞)`, clamped to `[0, 1]` in composite.
- **UTMOS** via [`fakerybakery/utmos`](https://github.com/fakerybakery/utmos). Range `[1, 5]`,
  min-max normalized to `[0, 1]` in composite.
- **speaker_sim** via SpeechBrain ECAPA-TDNN cosine similarity. Range `[-1, 1]`, mapped to
  `[0, 1]` in composite. Optional; off by default.

Default composite weights: `0.6 * (1 - WER) + 0.4 * norm(UTMOS)`. For voice-cloning runs we'll
add a `speaker_sim` term (planned weight 0.3, with the other two renormalized).

## Project structure

```
.
├── README.md                     this file
├── pyproject.toml                pinned deps (uv-managed)
├── run.sh                        convenience wrapper: ./run.sh smoke, etc.
├── config/                       per-method YAMLs (added in Phases 2-4)
├── data/
│   └── hard_prompts.txt          curated stress-test prompts
├── src/
│   ├── rewards/                  WER, UTMOS, speaker_sim, composite
│   ├── data/                     dataset generation/loading (Phase 2)
│   ├── methods/                  dpo.py, kto.py, grpo.py (Phases 2-4)
│   ├── eval/                     held-out eval (Phase 5)
│   └── utils/                    logging, checkpointing, LoRA helpers
├── scripts/
│   ├── 00_smoke_test_rewards.py     Phase 1
│   ├── 00b_check_vllm_compat.py     Phase 1
│   ├── 01_generate_dataset.py       Phase 2 (TBD)
│   ├── 02_train_dpo.py              Phase 2 (TBD)
│   ├── 03_train_kto.py              Phase 3 (TBD)
│   ├── 04_train_grpo.py             Phase 4 (TBD)
│   └── 05_evaluate_all.py           Phase 5 (TBD)
├── results/                      eval tables, sample audio
└── runs/                         training outputs (gitignored)
```

## Notes on determinism

All scripts set seeds for `random`, `numpy`, and `torch`. Runs are deterministic on the same
hardware/CUDA version but **may differ across GPU types** (A100 vs H100) due to kernel and
reduction-order differences. The seed is logged to wandb alongside other config.
