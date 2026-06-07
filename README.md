# TTSAlign: Reinforcement Fine-Tuning and Preference Optimization (DPO, KTO, GRPO) for Text-to-Speech

Two offline preference-optimization methods (DPO, KTO) and one online RL fine-tuning method (GRPO), applied to [OuteAI/Llama-OuteTTS-1.0-1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) and evaluated on 30 held-out prompts against the base model. Reward is a deterministic WER + UTMOS + ECAPA-speaker-similarity composite.

## Results

n=30 held-out eval at temperature 0.7, the sampling regime typical for expressive TTS output. Same reward pipeline as training. LoRA r=16/α=32 on attention + MLP across all three methods; DPO/KTO at 300 training steps, GRPO at 50 steps with K=8 rollouts per prompt; β=0.1 for DPO/KTO, β=0.05 for GRPO.

| method | mean WER | mean UTMOS | mean speaker_sim | mean composite | catastrophic failure rate (WER > 30%) |
|---|---|---|---|---|---|
| base | 0.068 | 4.36 | 0.580 | 0.871 | 13.3% |
| DPO (212 pairs) | 0.073 | 4.39 | 0.617 | 0.875 | 10.0% |
| KTO | **0.049** | **4.40** | **0.623** | **0.887** | **6.7%** |
| GRPO | 0.062 | 4.37 | 0.589 | 0.876 | 13.3% |

- All three methods outperform base on composite. KTO leads, reducing WER by 28% relative (0.068 → 0.049) alongside UTMOS and speaker-similarity gains.
- DPO and GRPO achieve nearly identical composite scores (0.875 vs 0.876) through different mechanisms. DPO improves UTMOS and speaker similarity but regresses on WER; GRPO improves WER and UTMOS but matches base's catastrophic-failure rate.
- DPO's WER above base is notable. The composite still improves through UTMOS and speaker_sim gains, but DPO does not improve intelligibility in this run.

### Why KTO outperforms DPO here

DPO and KTO are usually comparable in the literature on text-preference tasks; here KTO wins on every metric. The first suspect is data quantity — KTO trains on ~140 binary-labeled candidates while the initial DPO run used 53 paired examples (a 2.5× gap). To rule that out, DPO was rerun at 212 pairs (top-2 × bottom-2 per prompt — same construction logic as KTO's quantiles, just paired).

**DPO at 212 pairs: composite 0.875. DPO at 53 pairs: 0.876.** Identical within noise. Data quantity does not explain KTO's lead.

The difference is the loss function:

- KTO's quantile thresholds drop the ambiguous middle. DPO's pair construction does not have this option, so noisier pairs train it in the wrong direction at full strength when noisy composite scores flip the ordering of two close-quality candidates.
- The 4 candidates per prompt at temps [0.7, 0.9, 1.0, 1.2] cluster within ~0.03 composite. DPO's balanced 212-pair set is dominated by these low-margin pairs. The original 53 pairs were the max-margin pairs; adding the rest contributed noise rather than signal.

DPO is limited by data quality here, not data quantity. KTO's loss design avoids this limitation.

### GRPO notes

GRPO is online RL: each step samples K rollouts from the current policy, scores them via the reward composite, computes group-relative advantages, and updates. Compared to DPO/KTO (offline preference optimization on a fixed dataset), GRPO has more hyperparameters and more failure modes to manage. The configuration used here:

- **Rollout temperature 0.7.** Slightly higher than OuteTTS's 0.4 development default, chosen to reflect typical expressive-TTS deployment where natural-sounding speech is preferred over highly deterministic output. GRPO samples rollouts at this temperature during training, so the policy is optimized for the same sampling conditions used at inference.
- **K=8 rollouts per prompt** with `scale_rewards=False` (Dr. GRPO). Group-relative advantages need within-group reward variance to give the policy gradient real signal. K=8 stabilizes the within-group baseline — each rollout's advantage is computed against a 7-sample mean instead of a noisy 3-sample mean. `scale_rewards=False` keeps the raw `R − mean` advantage rather than dividing by a tiny within-group std, which would amplify reward-measurement noise into full-magnitude ±1 advantages.
- **LoRA dropout = 0.0**. TRL's `GRPOTrainer` calls `model.generate` while the model is in `train()` mode, so any nonzero `lora_dropout` would fire on every rollout and add noise to the generated audio that doesn't reflect the inference-time policy. PEFT's default for online RL is 0.0 for this reason.

GRPO improves over base on WER, UTMOS, and speaker similarity, achieving a composite (0.876) essentially tied with DPO (0.875) through different mechanisms. The exception is catastrophic-failure rate: GRPO matches base at 13.3% while DPO and KTO both reduce it. Online RL is more data-intensive than offline preference optimization by design, and at 53 training prompts the offline methods extract more signal per prompt, which explains KTO's lead.

### Honest caveats

- n=30 is small. Composite differences below ~0.01 are within per-seed noise.
- Many epochs over a 53-prompt training set — some training-set memorization is inevitable at this dataset size.
- Online RL does not lead at this scale. KTO is ahead on every composite-contributing metric. GRPO improves over base but extracts less signal per prompt than the offline methods at 53 prompts — see the GRPO notes.
- Composite is a proxy. Whisper has WER, UTMOS is a learned MOS estimate, ECAPA is one of several reasonable speaker encoders. The methods optimize the composite's preferences, not human preferences.

### Sample audio and training curves

Hand-picked sample audio (the 3 prompts with the highest base WER, where preference optimization had the most room to improve) at [`results/samples/`](results/samples/). Each prompt directory contains `base.wav`, `dpo.wav`, `kto.wav`, `grpo.wav` plus the source text. Full audio under [`results/audio/<method>/`](results/audio/) on the rented box (gitignored — too big for the repo). Training curves: [DPO run](https://wandb.ai/jcui-projects-personal/tts-rl/runs/646c97ki) · [KTO run](https://wandb.ai/jcui-projects-personal/tts-rl/runs/6i6cxo7x) · [GRPO run](https://wandb.ai/jcui-projects-personal/tts-rl/runs/1ahqqqfr) (wandb).

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
- ~1.26B params (Llama 3.2 1B base + extended audio-codec vocabulary) fits comfortably on a single 24GB GPU with LoRA.

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

# 5. Train DPO with LoRA on 212 top-vs-bottom-half pairs (~45 min total, ~$1)
#    Whisper-aligning + DAC-encoding 424 audio files is the bulk of the time.
./run.sh dpo

# 6. Train KTO with LoRA on ~140 binary-labeled candidates (~25 min, ~$0.50)
./run.sh kto

# 7. Train GRPO online — samples fresh rollouts each step (~3-3.5 hours, ~$4)
./run.sh grpo

# 8. Held-out eval comparing base / DPO / KTO / GRPO (~50 min, ~$0.80)
./run.sh eval

# Optional diagnostics:
uv add 'vllm==0.6.4.post1'
./run.sh vllm-check        # vLLM compatibility
./run.sh roundtrip-check   # audio-token round-trip via the codec
./run.sh grpo-rollout      # reproduce a GRPO rollout outside TRL for sanity checks
```

Every training script accepts `--smoke-test` for a fast (~5–10 min) pipeline check (5 steps, batch size 1) before committing to a long run.

## Rented compute setup (RunPod)

Development is from a MacBook with no GPU. All training runs on a rented RunPod A100 40GB.
Local machine = code, git, wandb dashboard. Remote = training.

### One-time RunPod setup

1. Sign up at [runpod.io](https://runpod.io) and add a payment method ($10 is plenty to get started).
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
offline dataset scoring (in `01_generate_dataset.py`) and the held-out eval (in `05_evaluate_all.py`).

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
├── run.sh                        convenience wrapper: ./run.sh smoke, dataset, dpo, kto, grpo, eval
├── config/
│   ├── dpo.yaml                  DPO hyperparameters
│   ├── kto.yaml                  KTO hyperparameters
│   └── grpo.yaml                 GRPO hyperparameters
├── data/
│   ├── easy_prompts.txt          starter set of natural English prompts
│   ├── hard_prompts.txt          curated stress-test prompts
│   └── dataset.parquet           scored preference dataset (committed)
├── src/
│   ├── rewards/                  WER, UTMOS, speaker_sim, composite
│   ├── data/                     dataset loading + chosen/rejected + KTO labels
│   └── utils/                    seeding + LoRA config helpers
├── scripts/
│   ├── 01_generate_dataset.py       generate the scored preference dataset
│   ├── 02_train_dpo.py              DPO training
│   ├── 03_train_kto.py              KTO training
│   ├── 04_train_grpo.py             GRPO training
│   ├── 05_evaluate_all.py           held-out comparison eval
│   └── diagnostics/                 environment/codec/sampling sanity checks
│       ├── smoke_test_rewards.py
│       ├── check_vllm_compat.py
│       ├── check_audio_roundtrip.py
│       ├── check_grpo_rollout.py
│       └── pick_samples.py
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
