"""
PerconAI — TinyHAR-Net Model Architecture
==========================================
Frozen architecture specification:
  Input:  (batch, channels, 128)
  Block1: Conv1d(in, 32, k=5, p=2) → BN(32) → ReLU → MaxPool(2)
  Block2: Conv1d(32, 64, k=5, p=2) → BN(64) → ReLU → MaxPool(2)
  Block3: Conv1d(64, 64, k=5, p=2) → BN(64) → ReLU → MaxPool(2)
  Block4: Conv1d(64, 64, k=5, p=2) → BN(64) → ReLU
  Head:   AdaptiveAvgPool1d(16) → Flatten → Linear(64*16,64) → ReLU → Linear(64, C)

  Constraint: total params < 150,000
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("PerconAI.model")


class ConvBlock(nn.Module):
    """Conv1d → BatchNorm1d → ReLU [→ MaxPool1d]."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        pool: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyHARNet(nn.Module):
    """
    TinyHAR-Net — lightweight 1-D CNN for wearable HAR.

    Parameters
    ----------
    in_channels : int
        Number of sensor channels (e.g., 9 for acc+gyro+mag XYZ).
    num_classes : int
        Number of activity classes.
    gap_size : int
        Output size of the global average-pooling layer (default 16).
    fc_hidden : int
        Hidden dimension of the classification head (default 64).
    max_params : int
        Hard ceiling on total parameter count (default 150 000).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        gap_size: int = 16,
        fc_hidden: int = 64,
        max_params: int = 150_000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # ── Feature extractor ────────────────────────────────────────────
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, pool=True),   # Block 1
            ConvBlock(32, 64, pool=True),             # Block 2
            ConvBlock(64, 64, pool=True),             # Block 3
            ConvBlock(64, 64, pool=False),            # Block 4 (no pool)
        )

        # ── Classification head ──────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * gap_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, num_classes),
        )

        # ── Parameter count assertion ────────────────────────────────────
        total = sum(p.numel() for p in self.parameters())
        assert total < max_params, (
            f"TinyHAR-Net has {total:,} params — exceeds ceiling of {max_params:,}. "
            f"Review architecture."
        )
        logger.info(
            "TinyHAR-Net created: in_ch=%d, classes=%d, params=%s (%.1f KB fp32)",
            in_channels, num_classes, f"{total:,}", total * 4 / 1024,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, channels, 128)

        Returns
        -------
        logits : Tensor of shape (batch, num_classes)
        """
        x = self.features(x)
        x = self.gap(x)
        return self.head(x)

    def get_bn_layers(self) -> list[nn.BatchNorm1d]:
        """Return all BatchNorm1d layers (used by BN-RECAL / TENT)."""
        return [m for m in self.modules() if isinstance(m, nn.BatchNorm1d)]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def estimate_sram_bytes(in_channels: int, num_classes: int, dtype: str = "fp32") -> int:
        """
        Rough estimate of peak SRAM during inference (weights + activations).
        Useful for device-profile feasibility checks.
        """
        model = TinyHARNet(in_channels, num_classes)
        param_bytes = sum(p.numel() for p in model.parameters()) * (4 if dtype == "fp32" else 1)
        # Activations: largest intermediate is after Block1 → (1, 32, 64) float32
        act_bytes = 32 * 64 * (4 if dtype == "fp32" else 1)
        return param_bytes + act_bytes


def build_tinyhar_net(cfg: dict, in_channels: int, num_classes: int) -> TinyHARNet:
    """Factory function that reads frozen config to build the model."""
    mcfg = cfg["model"]
    return TinyHARNet(
        in_channels=in_channels,
        num_classes=num_classes,
        gap_size=mcfg["gap_output_size"],
        fc_hidden=mcfg["fc_hidden"],
        max_params=mcfg["max_params"],
    )
