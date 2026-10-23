from __future__ import annotations

import math

import torch
import torch.nn as nn


def _mlp(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_out))


class LatentCommunicationLayer(nn.Module):
    """One round of latent communication between routing capsules R and model capability tokens A.

    1. Each model token reads the capsules through model-conditioned attention and is updated from
       [a_i; r_hat_i; a_i * r_hat_i].
    2. Model tokens compare with each other through availability-masked self-attention with the pairwise
       bias b_ij = psi([a_i; a_j; a_i - a_j; a_i * a_j]).
    3. Capsules receive feedback from the model tokens weighted by
       omega_i = softmax_i(-(max_j s_j - s_i) / tau).
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, feedback_temperature: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        h = int(hidden_dim)
        self.hidden_dim, self.num_heads, self.head_dim = h, int(num_heads), h // int(num_heads)
        self.feedback_temperature = float(feedback_temperature)
        self.dropout = nn.Dropout(dropout)

        self.read_key = nn.Linear(h, h)
        self.read_value = nn.Linear(h, h)
        self.read_query = nn.Linear(h, h)
        self.read_update = _mlp(3 * h, 2 * h, h, dropout)

        self.pair_qkv = nn.Linear(h, 3 * h)
        self.pair_bias = _mlp(4 * h, h, self.num_heads, dropout)
        self.pair_out = nn.Linear(h, h)

        self.feedback_score = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))
        self.feedback_query = nn.Linear(h, h)
        self.feedback_key = nn.Linear(h, h)
        self.feedback_value = nn.Linear(h, h)
        self.feedback_out = nn.Linear(h, h)

        self.model_ffn = _mlp(h, 4 * h, h, dropout)
        self.capsule_ffn = _mlp(h, 4 * h, h, dropout)
        self.norm_read = nn.LayerNorm(h)
        self.norm_pair = nn.LayerNorm(h)
        self.norm_model_ffn = nn.LayerNorm(h)
        self.norm_feedback = nn.LayerNorm(h)
        self.norm_capsule_ffn = nn.LayerNorm(h)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, capsules: torch.Tensor, models: torch.Tensor, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.read_query(models)
        k = self.read_key(capsules)
        v = self.read_value(capsules)
        read = torch.softmax(torch.einsum("bkd,bcd->bkc", q, k) / math.sqrt(float(self.hidden_dim)), dim=-1)
        r_hat = torch.einsum("bkc,bcd->bkd", read, v)
        models = self.norm_read(models + self.dropout(self.read_update(torch.cat([models, r_hat, models * r_hat], dim=-1))))
        models = torch.where(available[:, :, None], models, torch.zeros_like(models))

        qh, kh, vh = (self._heads(t) for t in self.pair_qkv(models).chunk(3, dim=-1))
        logits = torch.einsum("bhid,bhjd->bhij", qh, kh) / math.sqrt(float(self.head_dim))
        k_models = models.shape[1]
        ai = models[:, :, None, :].expand(-1, -1, k_models, -1)
        aj = models[:, None, :, :].expand(-1, k_models, -1, -1)
        logits = logits + self.pair_bias(torch.cat([ai, aj, ai - aj, ai * aj], dim=-1)).permute(0, 3, 1, 2)
        attn = torch.softmax(logits.masked_fill(~available[:, None, None, :], -1e9), dim=-1)
        context = torch.einsum("bhij,bhjd->bhid", attn, vh).transpose(1, 2).contiguous().view(models.shape[0], k_models, self.hidden_dim)
        models = self.norm_pair(models + self.dropout(self.pair_out(context)))
        models = self.norm_model_ffn(models + self.dropout(self.model_ffn(models)))
        models = torch.where(available[:, :, None], models, torch.zeros_like(models))

        scores = self.feedback_score(models).squeeze(-1).masked_fill(~available, -1e9)
        omega = torch.softmax(scores / max(self.feedback_temperature, 1e-6), dim=1)
        weighted = models * omega[:, :, None]
        fq = self._heads(self.feedback_query(capsules))
        fk = self._heads(self.feedback_key(weighted))
        fv = self._heads(self.feedback_value(weighted))
        logits = torch.einsum("bhcd,bhkd->bhck", fq, fk) / math.sqrt(float(self.head_dim))
        attn = torch.softmax(logits.masked_fill(~available[:, None, None, :], -1e9), dim=-1)
        context = torch.einsum("bhck,bhkd->bhcd", attn, fv).transpose(1, 2).contiguous().view(capsules.shape[0], capsules.shape[1], self.hidden_dim)
        capsules = self.norm_feedback(capsules + self.dropout(self.feedback_out(context)))
        capsules = self.norm_capsule_ffn(capsules + self.dropout(self.capsule_ffn(capsules)))
        return capsules, models


class LatentRouterNet(nn.Module):
    """Routing capsules, model capability tokens, H latent communication layers, the outcome head and the
    bounded capsule correction."""

    def __init__(self, query_dim: int, model_dim: int, hidden_dim: int = 96, num_query_tokens: int = 6,
                 num_capsules: int = 7, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 residual_bound: float = 0.05, feedback_temperature: float = 0.25) -> None:
        super().__init__()
        h = int(hidden_dim)
        self.hidden_dim, self.num_query_tokens = h, int(num_query_tokens)
        self.residual_bound = float(residual_bound)
        self.query_tokens = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, h * self.num_query_tokens))
        self.capsule_queries = nn.Parameter(torch.randn(num_capsules, h) * 0.02)
        self.capsule_attn = nn.MultiheadAttention(h, num_heads, dropout=float(dropout), batch_first=True)
        self.model_proj = _mlp(model_dim, h, h, dropout)
        self.layers = nn.ModuleList([LatentCommunicationLayer(h, num_heads, dropout, feedback_temperature) for _ in range(num_layers)])
        self.outcome_head = _mlp(h, h, 1, dropout)
        self.capsule_correction = _mlp(2 * h, h, 1, dropout)
        nn.init.zeros_(self.outcome_head[-1].weight)
        nn.init.zeros_(self.outcome_head[-1].bias)

    def forward(self, query: torch.Tensor, model_features: torch.Tensor, available: torch.Tensor) -> dict[str, torch.Tensor]:
        b = query.shape[0]
        tokens = self.query_tokens(query).view(b, self.num_query_tokens, self.hidden_dim)
        capsules, _ = self.capsule_attn(self.capsule_queries.unsqueeze(0).expand(b, -1, -1), tokens, tokens, need_weights=False)
        models = torch.where(available[:, :, None], self.model_proj(model_features), torch.zeros(1, device=query.device))
        for layer in self.layers:
            capsules, models = layer(capsules, models, available)

        prior = model_features[..., -1].clamp(1e-3, 1 - 1e-3)
        logit = self.outcome_head(models)[..., 0] + (torch.log(prior) - torch.log1p(-prior))
        mu = torch.sigmoid(logit)
        sigma = torch.sqrt(mu * (1 - mu) + 1e-6)
        pooled = capsules.mean(dim=1, keepdim=True).expand_as(models)
        delta = self.residual_bound * torch.tanh(self.capsule_correction(torch.cat([models, pooled], dim=-1)).squeeze(-1))
        mu_tilde = (mu + delta).clamp(0.0, 1.0)
        return {
            "logit": logit,
            "mu": mu.masked_fill(~available, -1e6),
            "sigma": sigma.masked_fill(~available, 1.0),
            "delta": delta.masked_fill(~available, 0.0),
            "mu_tilde": mu_tilde.masked_fill(~available, -1e6),
        }
