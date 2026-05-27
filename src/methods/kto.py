"""From-scratch KTO loss — for reference and learning.

Not used in the production training run (see scripts/03_train_kto.py, which
uses trl.KTOTrainer). This file shows what KTO actually computes.

The KTO paper (Ethayarajh et al., 2024):
    https://arxiv.org/abs/2402.01306

How KTO differs from DPO:
    - DPO needs *paired* (chosen, rejected) examples sharing one prompt.
    - KTO needs only *individual* (prompt, completion, binary_label) examples.
      Each label is True ("desirable") or False ("undesirable").
    - Real-world feedback often arrives as thumbs-up/thumbs-down on individual
      generations, not pairs. KTO matches that shape natively.
    - KTO's loss is *asymmetric*: it penalizes undesirable samples with high
      implicit reward more harshly than it rewards desirable samples with high
      implicit reward. This is borrowed from Kahneman-Tversky prospect theory —
      "losses loom larger than gains" — and is what makes the method robust to
      noisy labels.

The loss (per sample):
    pi_logratio = log pi(y|x) - log pi_ref(y|x)         # implicit reward
    kl_estimate = E_y~pi[log pi(y|x) - log pi_ref(y|x)]  # batch-level KL estimate
    if desirable:
        L = sigmoid(-beta * (pi_logratio - kl_estimate))
    else:
        L = sigmoid(-beta * (kl_estimate - pi_logratio))

Intuition: for a desirable sample, we want pi_logratio > kl_estimate (the
policy assigns more probability than the reference). For an undesirable one,
we want the opposite. sigmoid(-x) is small when x is large positive, so the
loss is small when the policy is moving in the right direction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kto_loss(
    policy_logps: torch.Tensor,    # (B,) sum_t log pi_theta(y_t | x, y_<t) for completion tokens
    ref_logps: torch.Tensor,       # (B,) same under the frozen reference policy
    labels: torch.Tensor,          # (B,) bool tensor: True iff sample is desirable
    kl_estimate: torch.Tensor,     # scalar: E_y~pi [log pi - log pi_ref], estimated on unrelated samples
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute KTO loss and diagnostics.

    `kl_estimate` is intentionally detached: it should be a held-out estimate
    of the policy's KL to the reference, not a quantity we backprop through.
    In TRL's implementation it's computed on the "unmatched" prompts shuffled
    against unrelated completions, which keeps it decoupled from the current
    sample's gradient.
    """
    pi_logratio = policy_logps - ref_logps
    kl = kl_estimate.detach() if kl_estimate.requires_grad else kl_estimate

    # Asymmetric prospect-theory loss
    desirable_mask = labels
    undesirable_mask = ~labels

    if desirable_mask.any():
        # Want pi_logratio > kl: sigmoid(-beta * (pi_logratio - kl)) -> 0
        l_des = 1.0 - torch.sigmoid(beta * (pi_logratio[desirable_mask] - kl))
        loss_desirable = l_des.mean()
    else:
        loss_desirable = torch.tensor(0.0, device=policy_logps.device)

    if undesirable_mask.any():
        # Want pi_logratio < kl: sigmoid(-beta * (kl - pi_logratio)) -> 0
        l_und = 1.0 - torch.sigmoid(beta * (kl - pi_logratio[undesirable_mask]))
        loss_undesirable = l_und.mean()
    else:
        loss_undesirable = torch.tensor(0.0, device=policy_logps.device)

    loss = desirable_weight * loss_desirable + undesirable_weight * loss_undesirable

    metrics = {
        "loss": loss.item(),
        "loss/desirable": loss_desirable.item(),
        "loss/undesirable": loss_undesirable.item(),
        "rewards/desirable": (beta * pi_logratio[desirable_mask]).mean().item() if desirable_mask.any() else 0.0,
        "rewards/undesirable": (beta * pi_logratio[undesirable_mask]).mean().item() if undesirable_mask.any() else 0.0,
        "kl_estimate": float(kl) if kl.dim() == 0 else float(kl.mean()),
    }
    return loss, metrics
