#!/usr/bin/env python3
# ============================================================================
#  models_revkd.py ― Reverse‑KD  Teacher & Student  (log‑anomaly detection)
#  (c) 2025  logsd project   MIT License
# ============================================================================

from __future__ import annotations
import math
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Positional Encoding
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sine–cosine positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5_000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # (B,S,D)
        return self.dropout(x + self.pe[:, : x.size(1)])


# --------------------------------------------------------------------------- #
#  Minimal block that can return attention weights
# --------------------------------------------------------------------------- #
class TransformerBlockWithAttn(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 ff_dim: int, dropout: float = 0.1):
        super().__init__()
        # ⚠️ need_weights 제거
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True                 # OK in ≥1.9
        )
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, need_attn: bool = False):
        # forward 에서만 need_weights 전달
        if need_attn:
            attn_out, attn_w = self.self_attn(
                x, x, x, need_weights=True               # ← 여기만 True
            )
        else:
            attn_out, attn_w = self.self_attn(
                x, x, x, need_weights=False
            )
            attn_w = None

        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_w


# --------------------------------------------------------------------------- #
#  Teacher network  (frozen during Reverse‑KD)
#  ‑ Only change:  `return_attn` flag to expose last‑layer attention matrix
# --------------------------------------------------------------------------- #
class TeacherNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 4096,
        dropout: float = 0.1,
        max_len: int = 5_000,
    ):
        super().__init__()

        self.pos = PositionalEncoding(embed_dim, max_len, dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockWithAttn(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.recon = nn.Linear(embed_dim, embed_dim)

    def forward(                       # (B,S,D_in)  →  (B,S,768)  (+ optional attn)
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        x = self.pos(x)
        last_attn = None
        for blk in self.blocks:
            x, a = blk(x, need_attn=return_attn)
            if return_attn:
                last_attn = a                          # (B,h,S,S)
        out = self.recon(x)
        return (out, last_attn) if return_attn else out


# --------------------------------------------------------------------------- #
#  Reverse‑KD Student  (Teacher output → Student → Original embedding space)
# --------------------------------------------------------------------------- #
class RevStudent(nn.Module):
    """
    3‑layer lightweight Transformer mapping T‑output back to embed‑space.

    Parameters
    ----------
    hidden_dim  : inner dim (default 384)
    num_heads   : attention heads (default 4)
    num_layers  : encoder layers (default 3)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.pos = PositionalEncoding(hidden_dim, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockWithAttn(
                    hidden_dim, num_heads, 4 * hidden_dim, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:            # (B,S,768)
        x = self.pos(self.in_proj(x))                              # (B,S,H)
        for blk in self.blocks:
            x, _ = blk(x)
        return self.out_proj(x)                                    # (B,S,768)


# --------------------------------------------------------------------------- #
#  Helper utilities
# --------------------------------------------------------------------------- #
def freeze_module(m: nn.Module) -> None:
    m.requires_grad_(False)
    m.eval()

def load_teacher_ckpt(
    path: str | Path,
    device: Optional[torch.device | str] = None,
    **kwargs,
) -> TeacherNet:
    teacher = TeacherNet(**kwargs)
    state = torch.load(str(path), map_location=device or "cpu")
    teacher.load_state_dict(state, strict=True)
    freeze_module(teacher)
    if device is not None:
        teacher.to(device)
    return teacher


# --------------------------------------------------------------------------- #
#  Public exports
# --------------------------------------------------------------------------- #
__all__ = [
    "PositionalEncoding",
    "TeacherNet",
    "RevStudent",
    "load_teacher_ckpt",
    "freeze_module",
]
