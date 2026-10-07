from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.rr_feature_development import HORIZON_MINUTES, build_origin_frame


def _contract() -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "config" / "public_feature_contract.yml").read_text())


def test_targets_are_future_demand_minus_origin_demand() -> None:
    rows = 2_030
    ts = pd.date_range("2015-04-01", periods=rows, freq="5min")
    frame = pd.DataFrame(
        {
            "SETTLEMENTDATE": ts,
            "INTERVENTION": 0,
            "TOTALDEMAND": range(rows),
            "DEMANDFORECAST": range(rows),
            "AVAILABLEGENERATION": 10_000.0,
            "NETINTERCHANGE": 0.0,
            "UIGF": 500.0,
            "is_weekend": 0,
            "is_public_holiday_qld": 0,
            "is_school_holiday_qld": 0,
            "total_qld_population": 5_000_000,
        }
    )
    out = build_origin_frame(frame, _contract())
    for minutes in HORIZON_MINUTES:
        assert out.loc[2016, f"target_t_plus_{minutes}m"] == minutes // 5
    assert bool(out.loc[2016, "eligible_common"])


def test_demandforecast_is_never_admitted() -> None:
    contract = _contract()
    admitted = {
        feature
        for group in contract["feature_groups"].values()
        for feature in group["features"]
    }
    assert "DEMANDFORECAST" not in admitted
    assert "DEMANDFORECAST" in contract["forbidden_model_inputs"]
