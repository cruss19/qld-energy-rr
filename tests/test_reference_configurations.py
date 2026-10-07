from __future__ import annotations

from pathlib import Path

import yaml


EXPECTED = {
    "rr_ref_01_persistence_30m_all_folds",
    "rr_ref_02_weekly_seasonal_naive_30m_all_folds",
    "rr_ref_03_comparable_history_30m_all_folds",
    "rr_ref_04_bayesian_random_walk_30m_all_folds",
    "rr_ref_05_original_ridge_30m_all_folds",
    "rr_ref_06_gaussian_linear_30m_all_folds",
    "rr_ref_07_student_t_linear_30m_all_folds",
    "rr_ref_08_empirical_residual_30m_all_folds",
    "rr_ref_09_aemo_p5min_all_folds",
}


def load_jobs(root: Path) -> list[dict]:
    paths = sorted((root / "config" / "experiments").glob("reference_[0-9][0-9]_*.yml"))
    return [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]


def test_nine_established_reference_jobs_are_declared() -> None:
    root = Path(__file__).resolve().parents[1]
    jobs = load_jobs(root)
    assert len(jobs) == 9
    assert {job["job_id"] for job in jobs} == EXPECTED


def test_original_references_are_30_minute_only() -> None:
    root = Path(__file__).resolve().parents[1]
    jobs = load_jobs(root)
    original = [job for job in jobs if job["model_name"] != "aemo_p5min_external_benchmark"]
    assert {job["folds_from"] for job in original} == {"config/experiments/reference_common.yml"}
    assert {job["major_parameters"]["horizon_minutes"] for job in original} == {30}
    assert {job["feature_contract_id"] for job in original} == {"rr_original_ridge_55_feature_contract_v1"}


def test_aemo_remains_six_horizon() -> None:
    root = Path(__file__).resolve().parents[1]
    job = next(job for job in load_jobs(root) if job["model_name"] == "aemo_p5min_external_benchmark")
    assert job["folds_from"] == "config/experiments/reference_aemo_common.yml"
    assert job["major_parameters"]["horizons_minutes"] == [5, 10, 15, 20, 25, 30]


def test_original_ridge_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    ridge = next(job for job in load_jobs(root) if job["model_name"] == "ridge_regression")
    assert ridge["variant"] == "original_55_feature_30m"
    assert ridge["major_parameters"]["base_predictors"] == 55
    assert ridge["major_parameters"]["solver"] == "sparse_cg"
    assert len(ridge["major_parameters"]["alpha_grid"]) == 17


def test_protocol_keeps_2020_out_of_development() -> None:
    root = Path(__file__).resolve().parents[1]
    for filename in ("reference_common.yml", "reference_aemo_common.yml"):
        common = yaml.safe_load((root / "config" / "experiments" / filename).read_text(encoding="utf-8"))
        folds = {fold["fold_id"]: fold for fold in common["folds"]}
        assert folds["validate_2019"]["train_end"] == "2018-12-31 23:55:00"
        assert folds["evaluate_2020"]["train_end"] == "2019-12-31 23:55:00"
        assert common["dataset"]["forbidden_learned_input"] == "DEMANDFORECAST"
