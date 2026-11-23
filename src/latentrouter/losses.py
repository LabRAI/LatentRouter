from __future__ import annotations

import torch
import torch.nn.functional as F


def outcome_nll(logit: torch.Tensor, quality: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    mask = available & torch.isfinite(quality)
    if not mask.any():
        return logit.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(logit[mask], quality[mask])


def pairwise_loss(scores: torch.Tensor, utility: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    valid = available & torch.isfinite(utility)
    pairs = valid[:, :, None] & valid[:, None, :]
    pairs = pairs & ~torch.eye(scores.shape[1], dtype=torch.bool, device=scores.device)[None]
    if not pairs.any():
        return scores.new_tensor(0.0)
    delta = scores[:, :, None] - scores[:, None, :]
    preferred = (utility[:, :, None] > utility[:, None, :]).float()
    return F.binary_cross_entropy_with_logits(delta[pairs], preferred[pairs])


def listwise_loss(scores: torch.Tensor, utility: torch.Tensor, available: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    target = torch.softmax(utility.masked_fill(~available, -1e6) / max(float(temperature), 1e-6), dim=1)
    log_probs = torch.log_softmax(scores.masked_fill(~available, -1e6), dim=1)
    loss = -(target * log_probs).sum(dim=1)
    rows = available.any(dim=1)
    return loss[rows].mean() if rows.any() else scores.new_tensor(0.0)


def ranking_losses(scores: torch.Tensor, utility: torch.Tensor, available: torch.Tensor, temperature: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    scores = scores.masked_fill(~available, -1e6)
    return pairwise_loss(scores, utility, available), listwise_loss(scores, utility, available, temperature)


def regression_loss(scores: torch.Tensor, targets: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    mask = available & torch.isfinite(targets)
    if not mask.any():
        return scores.new_tensor(0.0)
    return F.mse_loss(scores[mask], targets[mask])
