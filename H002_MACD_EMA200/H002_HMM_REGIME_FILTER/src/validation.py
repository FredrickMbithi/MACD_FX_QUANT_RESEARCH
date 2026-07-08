from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_folds(
    timestamps: pd.Series,
    train_years: int,
    test_months: int,
    step_months: int,
) -> list[WalkForwardFold]:
    ts = pd.to_datetime(timestamps).sort_values()
    start = ts.min()
    end = ts.max()

    folds = []
    fold_id = 1
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        train_start = train_start + pd.DateOffset(months=step_months)
    return folds


def split_by_fold(frame: pd.DataFrame, fold: WalkForwardFold, timestamp_col: str = "time_utc") -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(frame[timestamp_col])
    train = frame[(ts >= fold.train_start) & (ts < fold.train_end)].copy()
    test = frame[(ts >= fold.test_start) & (ts < fold.test_end)].copy()
    return train, test
