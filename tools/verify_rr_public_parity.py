"""Verify the RR public frame against the frozen reconstruction cache."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    public_path = root / "data/processed/rr_original_model_frame_5min_2015_2020.parquet"
    frozen_path = root / "training_output/private_cache/rr_original_model_frame_2015_2020.parquet"
    public = pd.read_parquet(public_path).set_index("datetime")
    frozen = pd.read_parquet(frozen_path).set_index("datetime")
    shared_rows = public.index.intersection(frozen.index)
    shared_columns = [column for column in frozen.columns if column in public.columns]
    if not shared_rows.equals(public.index):
        raise RuntimeError("The public frame contains timestamps absent from the frozen frame")
    frozen_only = frozen.index.difference(public.index)
    expected_terminal = pd.date_range("2020-12-31 23:30", "2020-12-31 23:55", freq="5min")
    if not frozen_only.equals(expected_terminal):
        raise RuntimeError(f"Unexpected timestamp difference: {list(frozen_only)}")
    mismatches = []
    for column in shared_columns:
        left, right = public.loc[shared_rows, column], frozen.loc[shared_rows, column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = np.isclose(left.to_numpy(float), right.to_numpy(float), equal_nan=True, rtol=1e-10, atol=1e-10)
        else:
            equal = (left.astype("string").fillna("<NA>").to_numpy() == right.astype("string").fillna("<NA>").to_numpy())
        if not bool(np.all(equal)):
            mismatches.append((column, int((~equal).sum()), float((~equal).mean())))
    print(f"rows={len(shared_rows)} shared_columns={len(shared_columns)} mismatched_columns={len(mismatches)}")
    print("terminal_rows_excluded_for_complete_six_horizon_targets=6")
    for item in mismatches:
        print(item)
    if mismatches:
        raise RuntimeError("RR public feature parity failed")
    print("PASS: all shared frozen feature columns are exactly reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
