"""From-scratch GRPO loss — for reference and learning.

Not used in the production training run (see scripts/04_train_grpo.py, which
uses trl.GRPOTrainer). This file shows what GRPO is actually computing.

GRPO is from DeepSeek-Math / DeepSeek-R1:
    https://arxiv.org/abs/2402.03300
    https://arxiv.org/abs/2501.12948

Key differences from DPO/KTO:
    - **Online**: rollouts are sampled from the *current* policy at every step,
      not from a fixed preference dataset.
    - **Reward function in the loop**: each rollout gets a scalar reward
      computed by an external function (for us: the WER + UTMOS + ECAPA composite).
    - **Group-relative advantages**: for each prompt, K rollouts are sampled.
      Each rollout's advantage is its (reward - group_mean) / group_std.
      This removes the need for a learned value/baseline network — the other
      rollouts on the *same prompt* are the baseline.

Loss (per token, per rollout, per group):
    advantage_i = (R_i - mean(R_group)) / std(R_group)
    ratio_t^i = pi_theta(o_t^i | x, o_<t^i) / pi_old(o_t^i | x, o_<t^i)
    L_pg = -mean over (i, t) of min(
        ratio_t^i * advantage_i,
        clip(ratio_t^i, 1-eps, 1+eps) * advantage_i
    )
    L_kl = beta * KL(pi_theta || pi_ref)
    L = L_pg + L_kl

This is PPO-style clipping with group-relative advantages instead of GAE.
TRL's GRPOTrainer implements this with all the masking, padding, and rollout
plumbing you'd otherwise have to write yourself.
"""
from __future__ import annotations

import torch


def group_relative_advantages(
    rewards: torch.Tensor,  # (B*K,) scalar reward per rollout
    group_size: int,        # K, rollouts per prompt
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute per-rollout advantages, normalized within each group of K rollouts.

    Reshapes to (B, K), z-scores along the K dimension, flattens back to (B*K,).
    A group of identical-reward rollouts yields advantages of zero (no learning
    signal), which is correct — there's no preference to learn from a flat group.
    """
    flat = rewards.view(-1, group_size)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True)
    advantages = (flat - mean) / (std + eps)
    return advantages.view(-1)


def grpo_loss(
    policy_logprobs: torch.Tensor,   # (B*K, T) per-token log prob under current policy
    old_logprobs: torch.Tensor,      # (B*K, T) per-token log prob at rollout time (frozen)
    ref_logprobs: torch.Tensor,      # (B*K, T) per-token log prob under reference policy
    advantages: torch.Tensor,        # (B*K,) group-relative advantage per rollout
    completion_mask: torch.Tensor,   # (B*K, T) 1 for completion tokens, 0 for prompt/padding
    beta: float = 0.05,
    clip_eps: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """PPO-clipped policy-gradient loss with KL penalty to the reference.

    All log-prob tensors are token-level. `advantages` is broadcast across the
    T dimension when computing the per-token loss.
    """
    # Importance-sampling ratio between current policy and the rollout-time policy.
    log_ratio = policy_logprobs - old_logprobs
    ratio = log_ratio.exp()

    # Broadcast advantages from (B*K,) to (B*K, T)
    adv = advantages.unsqueeze(-1)

    # PPO clipped surrogate
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_per_token = -torch.min(unclipped, clipped)

    # Token-level KL to reference, approximated as exp(diff) - diff - 1
    # (Schulman's "k3" estimator — non-negative, unbiased).
    kl_per_token = (ref_logprobs - policy_logprobs).exp() - (ref_logprobs - policy_logprobs) - 1.0

    per_token = pg_per_token + beta * kl_per_token
    # Mask out prompt + padding tokens, then mean over completion tokens.
    masked = per_token * completion_mask
    n_tokens = completion_mask.sum().clamp(min=1.0)
    loss = masked.sum() / n_tokens

    metrics = {
        "loss": loss.item(),
        "loss/pg": (pg_per_token * completion_mask).sum().item() / n_tokens.item(),
        "loss/kl": (kl_per_token * completion_mask).sum().item() / n_tokens.item(),
        "rewards/advantage_abs_mean": advantages.abs().mean().item(),
    }
    return loss, metrics
