from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.rr_feature_development import load_feature_contract


ROOT = Path(__file__).resolve().parents[1]


def test_public_parity_manifest_matches_current_contract() -> None:
    manifest = json.loads(
        (ROOT / "outputs/03_feature_contract/public_parity_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    contract, contract_hash = load_feature_contract(
        ROOT / "config/public_feature_contract.yml"
    )

    assert manifest["verification_status"] == "PASS"
    assert manifest["authoritative_contract_id"] == contract["contract_id"]
    assert manifest["authoritative_contract_sha256"] == contract_hash
    assert manifest["eligible_origin_count"] == 629_273
    assert manifest["compared_columns"] == 96
    assert manifest["mismatched_columns"] == 0
    assert manifest["frozen_comparison_artifact_distributed"] is False


def test_final_tcn_identity_manifest_is_complete_and_sanitized() -> None:
    path = ROOT / "outputs/model_artifacts/final_tcn_member_identities.csv"
    identities = pd.read_csv(path)

    assert len(identities) == 15
    assert set(identities["model_family"]) == {
        "TCN1",
        "TCN2",
        "TCN3",
        "TCN_star",
        "TCN_starNLL",
    }
    assert set(identities["seed"]) == {42, 142, 242}
    assert set(identities["member_status"]) == {"complete"}
    assert identities.groupby("model_family")["fixed_epochs"].first().to_dict() == {
        "TCN1": 7,
        "TCN2": 3,
        "TCN3": 5,
        "TCN_star": 10,
        "TCN_starNLL": 7,
    }
    assert not any(
        token in path.read_text(encoding="utf-8").lower()
        for token in ("c:\\users", "/home/", ".pt,", ".pth,", "prediction_path")
    )
