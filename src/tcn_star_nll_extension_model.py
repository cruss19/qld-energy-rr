"""TCN_starNLL widened only for the causal-estimate extension channels."""

from __future__ import annotations

import torch

from .tcn_star_nll_model import TCNStarNLL


class TCNStarNLLCausalFeatureEstimates(TCNStarNLL):
    """Original probabilistic architecture with the required input widths."""

    DEMAND_CHANNELS = 8
    REGIONAL_FEATURES = 12

    def _validate(self, demand, system, climate, regional, calendar, population):
        batch = demand.size(0)
        expected = {
            "demand": (batch, self.DEMAND_CHANNELS, self.SEQUENCE_STEPS),
            "system": (batch, self.SYSTEM_CHANNELS, self.SEQUENCE_STEPS),
            "climate": (batch, self.CLIMATE_CHANNELS, self.SEQUENCE_STEPS),
            "regional": (
                batch, self.REGIONS, self.REGIONAL_FEATURES, self.SEQUENCE_STEPS
            ),
            "calendar": (batch, self.calendar_dim),
            "population": (batch, 1),
        }
        actual = {
            "demand": tuple(demand.shape),
            "system": tuple(system.shape),
            "climate": tuple(climate.shape),
            "regional": tuple(regional.shape),
            "calendar": tuple(calendar.shape),
            "population": tuple(population.shape),
        }
        invalid = {name: (actual[name], shape) for name, shape in expected.items()
                   if actual[name] != shape}
        if invalid:
            raise ValueError(f"Extension tensor contract mismatch: {invalid}")
        return batch

    def forward(self, demand, system, climate, regional, calendar, population):
        batch = self._validate(demand, system, climate, regional, calendar, population)
        context = torch.cat([system, climate], dim=1)
        regional_flat = regional.contiguous().view(
            batch, self.REGIONS * self.REGIONAL_FEATURES, self.SEQUENCE_STEPS
        )
        fused = self.fusion_trunk(torch.cat([
            self.demand_encoder(demand)[:, :, -1],
            self.context_encoder(context)[:, :, -1],
            self.regional_encoder(regional_flat)[:, :, -1],
            calendar,
            population,
        ], dim=1))
        mu = self.mu_head(fused)
        scale = torch.nn.functional.softplus(self.scale_head(fused)) + self.minimum_scale
        df_fraction = self.DF_FRACTION_EPSILON + (
            1.0 - 2.0 * self.DF_FRACTION_EPSILON
        ) * torch.sigmoid(self.df_head(fused))
        df = self.minimum_df + (self.maximum_df - self.minimum_df) * df_fraction
        return mu, scale, df


__all__ = ["TCNStarNLLCausalFeatureEstimates"]
