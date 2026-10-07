"""Preserved public TCN_starNLL architecture for the RR reconstruction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import StudentT


class CausalConv1d(nn.Module):
    """Length-preserving 1-D convolution with strict left causality."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y[:, :, :-self.padding] if self.padding else y


class ResidualBlock(nn.Module):
    """Original two-convolution GELU/dropout residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        dropout: float = 0.33,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout1d(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout1d(dropout),
        )
        self.residual = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) + self.residual(x)


class TCNStarNLL(nn.Module):
    """Original public TCN_starNLL architecture and feature-routing contract."""

    DEMAND_CHANNELS = 7
    SYSTEM_CHANNELS = 16
    CLIMATE_CHANNELS = 2
    REGIONS = 5
    REGIONAL_FEATURES = 10
    SEQUENCE_STEPS = 505
    OUTPUT_HORIZONS = 6
    DF_FRACTION_EPSILON = 1e-4

    def __init__(
        self,
        calendar_dim: int,
        branch_out_dim: int = 32,
        dropout: float = 0.33,
        sequence_steps: int = 505,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32),
        minimum_scale: float = 1e-4,
        minimum_df: float = 2.1,
        maximum_df: float = 20.0,
        initial_df: float = 4.0,
    ) -> None:
        super().__init__()
        if sequence_steps != self.SEQUENCE_STEPS:
            raise ValueError(f"TCN_starNLL requires exactly {self.SEQUENCE_STEPS} steps")
        self.calendar_dim = int(calendar_dim)
        self.branch_out_dim = int(branch_out_dim)
        self.sequence_steps = int(sequence_steps)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(x) for x in dilations)
        self.minimum_scale = float(minimum_scale)
        self.minimum_df = float(minimum_df)
        self.maximum_df = float(maximum_df)
        self.demand_encoder = nn.Sequential(
            *self._encoder(self.DEMAND_CHANNELS, self.branch_out_dim, dropout)
        )
        self.context_encoder = nn.Sequential(
            *self._encoder(
                self.SYSTEM_CHANNELS + self.CLIMATE_CHANNELS,
                self.branch_out_dim,
                dropout,
            )
        )
        self.regional_encoder = nn.Sequential(
            *self._encoder(self.REGIONS * self.REGIONAL_FEATURES, 64, dropout)
        )
        fusion_width = 2 * self.branch_out_dim + 64 + self.calendar_dim + 1
        self.fusion_trunk = nn.Sequential(
            nn.Linear(fusion_width, 128), nn.GELU(), nn.LayerNorm(128)
        )
        self.mu_head = nn.Linear(128, self.OUTPUT_HORIZONS)
        self.scale_head = nn.Linear(128, self.OUTPUT_HORIZONS)
        self.df_head = nn.Linear(128, self.OUTPUT_HORIZONS)
        self._initialise_distribution_heads(initial_df)

    def _encoder(self, in_channels: int, out_channels: int, dropout: float):
        return [
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                dilation=dilation,
                dropout=dropout,
                kernel_size=self.kernel_size,
            )
            for i, dilation in enumerate(self.dilations)
        ]

    def _initialise_distribution_heads(self, initial_df: float) -> None:
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(
            self.scale_head.bias,
            math.log(math.expm1(1.0 - self.minimum_scale)),
        )
        nn.init.zeros_(self.df_head.weight)
        bounded = (initial_df - self.minimum_df) / (self.maximum_df - self.minimum_df)
        fraction = (bounded - self.DF_FRACTION_EPSILON) / (
            1.0 - 2.0 * self.DF_FRACTION_EPSILON
        )
        nn.init.constant_(self.df_head.bias, math.log(fraction / (1.0 - fraction)))

    @property
    def receptive_field_steps(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def _validate(self, demand, system, climate, regional, calendar, population):
        b = demand.size(0)
        expected = {
            "demand": (b, 7, 505), "system": (b, 16, 505),
            "climate": (b, 2, 505), "regional": (b, 5, 10, 505),
            "calendar": (b, self.calendar_dim), "population": (b, 1),
        }
        actual = {
            "demand": tuple(demand.shape), "system": tuple(system.shape),
            "climate": tuple(climate.shape), "regional": tuple(regional.shape),
            "calendar": tuple(calendar.shape), "population": tuple(population.shape),
        }
        invalid = {k: (actual[k], v) for k, v in expected.items() if actual[k] != v}
        if invalid:
            raise ValueError(f"TCN_starNLL tensor contract mismatch: {invalid}")
        return b

    def forward(self, demand, system, climate, regional, calendar, population):
        batch = self._validate(demand, system, climate, regional, calendar, population)
        context = torch.cat([system, climate], dim=1)
        regional_flat = regional.contiguous().view(batch, 50, 505)
        fused = self.fusion_trunk(torch.cat([
            self.demand_encoder(demand)[:, :, -1],
            self.context_encoder(context)[:, :, -1],
            self.regional_encoder(regional_flat)[:, :, -1],
            calendar,
            population,
        ], dim=1))
        mu = self.mu_head(fused)
        scale = nn.functional.softplus(self.scale_head(fused)) + self.minimum_scale
        df_fraction = self.DF_FRACTION_EPSILON + (
            1.0 - 2.0 * self.DF_FRACTION_EPSILON
        ) * torch.sigmoid(self.df_head(fused))
        df = self.minimum_df + (self.maximum_df - self.minimum_df) * df_fraction
        return mu, scale, df


class DynamicStudentTNLLLoss(nn.Module):
    def forward(self, pred_mu, pred_scale, pred_df, target):
        if not (pred_mu.shape == pred_scale.shape == pred_df.shape == target.shape):
            raise ValueError("mu, scale, df and target shapes must match")
        return -StudentT(df=pred_df, loc=pred_mu, scale=pred_scale).log_prob(target).mean()


__all__ = ["TCNStarNLL", "DynamicStudentTNLLLoss"]
