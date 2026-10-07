from pathlib import Path

import pandas as pd

from src.data_provenance import audit_manifest, load_yaml, source_inventory, validate_registry_policy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_yaml(ROOT / "config" / "data_acquisition.yml")
REGISTRY = load_yaml(ROOT / CONFIG["paths"]["source_registry_path"])
MANIFEST = ROOT / CONFIG["paths"]["manifest_path"]


def test_registry_rejects_post_2020_sources() -> None:
    validate_registry_policy(REGISTRY)
    assert pd.Timestamp(REGISTRY["policy"]["observation_end"]) == pd.Timestamp("2020-12-31")
    assert pd.Timestamp(REGISTRY["policy"]["latest_source_release"]) == pd.Timestamp("2020-12-31")


def test_every_declared_source_is_exactly_registered() -> None:
    inventory = source_inventory(ROOT, REGISTRY, MANIFEST)
    assert inventory["status"].eq("registered and verified").all(), inventory.to_string(index=False)


def test_manifest_has_no_post_2020_coverage() -> None:
    audit = audit_manifest(ROOT, MANIFEST)
    assert not audit.empty
    assert audit[["exists", "size_matches", "sha256_matches", "coverage_within_2020"]].all().all()


def test_no_later_edition_is_admitted() -> None:
    paths = "\n".join(pd.read_csv(MANIFEST)["relative_path"].astype(str)).lower()
    for forbidden in ("2021", "2022", "2023", "2024", "2025", "2026", "to-present"):
        assert forbidden not in paths
