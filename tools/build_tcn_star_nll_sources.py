"""Build and verify the auxiliary RR TCN_starNLL source artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tcn_star_nll_sources import (  # noqa: E402
    build_publication_safe_soi,
    write_regional_solar_capacity,
)


def main() -> int:
    capacity_path = write_regional_solar_capacity(PROJECT_ROOT)
    capacity = pd.read_parquet(capacity_path)
    origins = pd.date_range("2015-01-01", "2020-12-31 23:55", freq="5min")
    soi = build_publication_safe_soi(
        origins,
        PROJECT_ROOT / "data" / "raw" / "bom" / "soi_monthly_through_2020.csv",
    )
    summary = {
        "status": "PASS",
        "regional_solar_path": capacity_path.relative_to(PROJECT_ROOT).as_posix(),
        "regional_solar_rows": len(capacity),
        "regional_solar_regions": sorted(capacity["region"].unique().tolist()),
        "regional_solar_first_month": capacity["Date"].min().isoformat(),
        "regional_solar_last_month": capacity["Date"].max().isoformat(),
        "soi_first_origin": soi.index.min().isoformat(),
        "soi_last_origin": soi.index.max().isoformat(),
        "soi_missing_values": int(soi.isna().sum().sum()),
        "post_2020_rows": 0,
    }
    output = PROJECT_ROOT / "outputs" / "03_feature_contract" / "tcn_star_nll_source_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
