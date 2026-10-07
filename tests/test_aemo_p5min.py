from __future__ import annotations

import pandas as pd

from src.aemo_p5min import filter_contract_rows


def test_p5min_filter_keeps_six_horizons_and_hard_2020_boundary() -> None:
    origin = pd.Timestamp("2020-12-31 23:30:00")
    rows = []
    for minutes in (5, 10, 15, 20, 25, 30, 35):
        rows.append(
            {
                "RUN_DATETIME": origin,
                "INTERVAL_DATETIME": origin + pd.Timedelta(minutes=minutes),
                "REGIONID": "QLD1",
                "TOTALDEMAND": 7000.0 + minutes,
                "DEMANDFORECAST": 1.0,
                "LASTCHANGED": origin,
            }
        )
    result = filter_contract_rows(
        pd.DataFrame(rows),
        source_month="2020-12",
        source_url="https://example.invalid/aemo.zip",
        region_id="QLD1",
        horizons_minutes=(5, 10, 15, 20, 25, 30),
    )
    assert result["horizon_minutes"].tolist() == [5, 10, 15, 20, 25]
    assert result["target_timestamp"].max() <= pd.Timestamp("2020-12-31 23:55:00")


def test_p5min_filter_deduplicates_with_latest_lastchanged() -> None:
    frame = pd.DataFrame(
        {
            "RUN_DATETIME": [pd.Timestamp("2019-01-01 00:00:00")] * 2,
            "INTERVAL_DATETIME": [pd.Timestamp("2019-01-01 00:05:00")] * 2,
            "REGIONID": ["QLD1", "QLD1"],
            "TOTALDEMAND": [6000.0, 6100.0],
            "DEMANDFORECAST": [0.0, 0.0],
            "LASTCHANGED": [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-01 00:01")],
        }
    )
    result = filter_contract_rows(
        frame,
        source_month="2019-01",
        source_url="https://example.invalid/aemo.zip",
        region_id="QLD1",
        horizons_minutes=(5, 10, 15, 20, 25, 30),
    )
    assert len(result) == 1
    assert result.loc[0, "forecast_demand_mw"] == 6100.0
