from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.rr_feature_development import (
    HORIZON_MINUTES,
    build_origin_frame,
    expanded_contract_rows,
    load_feature_contract,
)


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


def test_expanded_contract_matches_authoritative_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    contract, digest = load_feature_contract(root / "config/public_feature_contract.yml")
    expected = expanded_contract_rows(contract, digest).fillna("")
    retained = pd.read_csv(
        root / "data/contracts/rr_public_feature_contract_v1.csv",
        keep_default_na=False,
    )
    pd.testing.assert_frame_equal(
        retained.astype(str), expected.astype(str), check_dtype=False
    )
    assert retained["contract_sha256"].eq(digest).all()
    assert not retained["route"].str.contains("flattened states", case=False).any()
    ridge = yaml.safe_load(
        (root / "config/experiments/reference_05_original_ridge.yml").read_text(
            encoding="utf-8"
        )
    )
    assert ridge["variant"] == "original_55_feature_30m"
    assert ridge["major_parameters"]["base_predictors"] == 55
