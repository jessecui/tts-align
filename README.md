# TTSAlign: Preference Optimization (DPO vs. KTO) for Autoregressive Text-to-Speech

Exploratory hobby project applying two preference-optimization methods — **DPO** and **KTO** —
to the same small autoregressive codec-LM TTS model ([OuteAI/Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B)),
with a held-out comparison eval against the base model.

**Status:** done. DPO and KTO both trained, evaluated against base on 30 held-out prompts. Results below.

## Results

30-prompt held-out eval. Same reward pipeline as training (Whisper-small WER, UTMOS, ECAPA-TDNN speaker similarity, composite). Both methods used identical LoRA setups (rank 16, alpha 32, attention + MLP projections), 300 training steps, beta 0.1.

| method | mean WER | mean UTMOS | mean speaker_sim | mean composite | catastrophic failure rate (WER > 30%) |
|---|---|---|---|---|---|
| base | 0.068 | 4.36 | 0.580 | 0.871 | 13.3% |
| DPO  | 0.061 | 4.38 | 0.586 | 0.876 | 10.0% |
| KTO  | **0.057** | **4.38** | **0.619** | **0.882** | **10.0%** |

Headlines (kept honest about n=30 — these are *directional* signals, not statistically defensible deltas):
- **Both methods beat the base model** on every metric in the table.
- **KTO edges DPO across the board.** Largest margin is on speaker similarity: +0.039 over base, +0.033 over DPO. The other deltas are smaller and within plausible per-seed noise.
- **Catastrophic failures dropped from 4/30 to 3/30 for both DPO and KTO** — directionally good, but a one-prompt difference at n=30.
- **KTO's mean WER came in 0.011 absolute below base** (0.057 vs 0.068). Suggestive, not significant — at n=30 with no confidence intervals you can't conclude more.

### Why KTO > DPO here — a counterintuitive result worth examining

Textbook intuition is that DPO should be more robust: paired comparisons cancel absolute-scale noise, and the loss has a clean closed-form RL derivation. In this setup KTO won on every metric. A few structural reasons:

- **KTO got more training items.** DPO uses only `best` vs `worst` per prompt → 53 pairs. KTO labels the top and bottom thirds of all candidates → ~140 examples. 2.5x more training items drawn from candidates DPO discarded.
- **KTO overfit less.** DPO ran 43 epochs over 53 pairs; KTO ran 17 epochs over 140 examples. Less repetition → better held-out generalization.
- **Our pairs are noisy.** We constructed pairs by sorting candidates on a noisy composite score (Whisper + UTMOS + ECAPA each have measurement error). When chosen and rejected are close in true quality, the sort can flip, and DPO trains in the wrong direction with full force. KTO's quantile thresholds skip the ambiguous middle entirely.
- **Within-prompt margins are usually small.** The 4 candidates per prompt at temps [0.7, 0.9, 1.0, 1.2] often score within 0.03 of each other. DPO is forced to declare a strong "this >> that" preference; KTO just sorts top-third vs bottom-third.

The "DPO is more robust" intuition comes from a setting with clean, human-labeled paired preferences. With synthetic preferences from an automatic reward model, KTO's structural advantages — more data, more noise tolerance, asymmetric loss — dominate.

Honest caveats:
- 30 prompts is small. No confidence intervals; differences between DPO and KTO at this scale could easily be one std-dev of noise.
- **The DPO-vs-KTO comparison isn't strictly apples-to-apples** as run here. KTO got ~2.5x more training items (140 vs 53) because we kept the top *and* bottom thirds, while DPO only paired best-vs-worst. The obvious followup is to re-run KTO on a 53-example subset, isolating the loss function from the data-quantity confound. Not done yet — flagging as future work.
- Both methods trained 16–43 epochs over a small dataset (53 train prompts × 4 candidates = 212 candidates), so some training-set memorization. The held-out numbers are what matter, but a larger dataset would give a stronger story.
- The reward pipeline is itself imperfect (Whisper makes mistakes; UTMOS is a learned approximation of human ratings; ECAPA is one of several reasonable speaker encoders). The "preferences" the methods are learning to satisfy are *the reward pipeline's preferences*, not human preferences directly.

Hand-picked sample audio (the 3 prompts where base WER was highest — i.e., the prompts preference optimization had the most room to help on) at [`results/samples/`](results/samples/). Each prompt directory contains `base.wav`, `dpo.wav`, `kto.wav` plus the source text. Full audio under [`results/audio/<method>/`](results/audio/) on the rented box (gitignored — too big for the repo). Training curves: [DPO run](https://wandb.ai/jcui-projects-personal/tts-rl/runs/m39opezt) · [KTO run](https://wandb.ai/jcui-projects-personal/tts-rl/runs/6i6cxo7x) (wandb).

## Plan

| Phase | Scope                                              | Status |
|-------|----------------------------------------------------|--------|
| 1     | Repo scaffold + reward pipeline + smoke test       | Done |
| 2     | Preference dataset generation + DPO end-to-end     | Done |
| 3     | KTO                                                | Done |
| 4     | GRPO (online; vLLM-accelerated sampling)           | Scoped out — see below |
| 5     | Held-out comparison eval + writeup                 | Done |

**Why GRPO was dropped from the original plan:** the original pitch was a three-way comparison (DPO/KTO/GRPO). I scoped down after seeing how much engineering the OuteTTS integration took — particularly the word-aligned codec-token training format, which doesn't match the "raw audio tokens" assumption DPO/KTO/GRPO infrastructure usually targets. GRPO adds another order of magnitude of complexity (online rollouts inside the training loop, reward computation per step) on top of that. The remaining DPO-vs-KTO comparison is still a meaningful demonstration of preference optimization on TTS, just not the broader DPO-vs-KTO-vs-GRPO comparison originally pitched. Future work — happy to revisit if the project picks up steam.

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

# 3. Verify the reward pipeline works on a few generated samples (~5-8 min)
#    Downloads OuteTTS-1.0-1B (~2GB), DAC decoder, Whisper-small (~500MB), UTMOS, ECAPA-TDNN.
./run.sh smoke --voice-cloning

# 4. Generate the scored preference dataset (~2-3 hours, ~$2-3 on A100)
#    Resumable — Ctrl-C and re-run, it picks up where it left off.
./run.sh dataset

# 5. Train DPO with LoRA (~20 min, ~$0.40)
./run.sh dpo

# 6. Train KTO with LoRA (~20 min, ~$0.40)
./run.sh kto

# 7. Held-out eval comparing base / DPO / KTO (~45 min, ~$0.80)
./run.sh eval

# Optional: verify vLLM compatibility — was on the path for GRPO, kept for future use
uv add 'vllm==0.6.4.post1'
./run.sh vllm-check
```

Every training script accepts `--smoke-test` for a fast (~5–10 min) pipeline check (5 steps, batch size 1) before committing to a long run.

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
offline dataset scoring (Phase 2) and the held-out eval (Phase 5).

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
  `[0, 1]` in composite. Off in the `CompositeWeights` default; enabled for the dataset and eval runs in this project.

Effective composite weights used throughout this project: **WER 0.46, UTMOS 0.31, speaker_sim 0.23** (configured in `src/rewards/composite.py` as `0.6 / 0.4 / 0.3` pre-normalization).

## Project structure

```
.
├── README.md                     this file
├── pyproject.toml                pinned deps (uv-managed)
├── run.sh                        convenience wrapper: ./run.sh smoke, dataset, dpo, kto, eval
├── config/
│   ├── dpo.yaml                  DPO hyperparameters
│   └── kto.yaml                  KTO hyperparameters
├── data/
│   ├── easy_prompts.txt          starter set of natural English prompts
│   ├── hard_prompts.txt          curated stress-test prompts
│   └── dataset.parquet           scored preference dataset (committed)
├── src/
│   ├── rewards/                  WER, UTMOS, speaker_sim, composite
│   ├── data/                     dataset loading + chosen/rejected + KTO labels
│   ├── methods/                  dpo.py, kto.py — from-scratch losses for reference
│   └── utils/                    seeding + LoRA config helpers
├── scripts/
│   ├── 00_smoke_test_rewards.py     Phase 1
│   ├── 00b_check_vllm_compat.py     Phase 1
│   ├── 01_generate_dataset.py       Phase 2
│   ├── 02_train_dpo.py              Phase 2
│   ├── 03_train_kto.py              Phase 3
│   └── 05_evaluate_all.py           Phase 5
├── results/
│   ├── audio/<method>/              sample audio per method
│   ├── comparison.md                comparison table
│   └── eval.parquet                 per-sample eval scores
└── runs/                         training checkpoints (gitignored)
```

## Notes on determinism

All scripts set seeds for `random`, `numpy`, and `torch`. Runs are deterministic on the same
hardware/CUDA version but **may differ across GPU types** (A100 vs H100) due to kernel and
reduction-order differences. The seed is logged to wandb alongside other config.
