"""From-scratch DPO loss — for reference and learning.

This file is NOT used in the production training run (see scripts/02_train_dpo.py,
which uses trl.DPOTrainer). It exists so you can read what DPO actually computes
without digging through TRL's training-loop scaffolding.

The DPO paper (Rafailov et al., 2023):
    https://arxiv.org/abs/2305.18290

Intuition:
    Vanilla RLHF: fit a reward model on preferences, then PPO with KL to a frozen
    reference. The reward model is the bottleneck (expensive, hard to evaluate,
    can be hacked). DPO observes that under the closed-form solution to the
    KL-regularized RL objective, the *implicit* reward is a known function of the
    policy and reference log-probs:
        r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) ) + const
    Substituting into the Bradley-Terry preference likelihood gives a loss that
    can be optimized directly on (prompt, chosen, rejected) triples — no separate
    reward model, no PPO, no rollouts. Just classification.

Loss:
    L = -log sigmoid( beta * [ (log pi(yc|x) - log pi_ref(yc|x))
                              - (log pi(yr|x) - log pi_ref(yr|x)) ] )

    yc = chosen, yr = rejected, beta = strength of KL (larger -> stays closer to ref).

This module assumes you've already computed the four log-prob sums (sum over
target tokens, masking out the prompt). TRL does that masking inside DPOTrainer;
here we just write the loss in five lines and ten lines of comments.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,   # (B,) sum_t log pi_theta(y_chosen_t | x, y_<t)
    policy_rejected_logps: torch.Tensor, # (B,) sum_t log pi_theta(y_rejected_t | x, y_<t)
    ref_chosen_logps: torch.Tensor,      # (B,) same under the frozen reference policy
    ref_rejected_logps: torch.Tensor,    # (B,) same
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Compute the DPO loss and a few useful diagnostics.

    All four inputs are per-example sums of token log-probabilities over the
    completion (target) tokens only. Prompt tokens are masked out upstream.
    """
    # log[ pi(yc|x) / pi_ref(yc|x) ] - log[ pi(yr|x) / pi_ref(yr|x) ]
    # The reference terms make this an *implicit* KL-regularized objective.
    pi_logratio_chosen = policy_chosen_logps - ref_chosen_logps
    pi_logratio_rejected = policy_rejected_logps - ref_rejected_logps
    margin_logits = pi_logratio_chosen - pi_logratio_rejected

    # Bradley-Terry NLL on the implicit reward differences.
    # logsigmoid is numerically stable; equivalent to -log(1 + exp(-beta * margin)).
    loss = -F.logsigmoid(beta * margin_logits).mean()

    # Diagnostics
    chosen_reward = beta * pi_logratio_chosen.detach()
    rejected_reward = beta * pi_logratio_rejected.detach()
    metrics = {
        "loss": loss.item(),
        "rewards/chosen": chosen_reward.mean().item(),
        "rewards/rejected": rejected_reward.mean().item(),
        "rewards/margin": (chosen_reward - rejected_reward).mean().item(),
        "rewards/accuracy": (chosen_reward > rejected_reward).float().mean().item(),
    }
    return loss, metrics
