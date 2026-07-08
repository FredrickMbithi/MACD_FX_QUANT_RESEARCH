from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

TIMESTAMP_COL = "time_utc"
OPEN_COL = "open"
HIGH_COL = "high"
LOW_COL = "low"
CLOSE_COL = "close"
VOLUME_COL = "tick_volume"

REQUIRED_COLUMNS = [TIMESTAMP_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]
PRICE_COLUMNS = [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]

# Bars with tick_volume at or below this are flagged as thin-liquidity,
# not wrong - see check_volume_sanity for why this is a WARNING.
LOW_VOLUME_THRESHOLD = 5

# A bar's high-low range is flagged as an outlier once its robust
# z-score (see check_price_outliers) exceeds this. 10 is deliberately
# conservative - real news-event bars can spike to 5-8x the median
# range, so this is aimed at catching mis-scaled/bad-tick bars, not
# ordinary volatility.
RANGE_OUTLIER_Z_THRESHOLD = 10

REPORT_TITLE = "MACD FX Quant Research"
REPORT_HYPOTHESIS = "H001 - MACD Baseline"
REPORT_MODULE = Path(__file__).name


@dataclass
class ValidationIssue:
    severity: str
    check: str
    message: str


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)
    rows_checked: int = 0

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "ERROR" for issue in self.issues)

    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        data_safe = "YES" if self.is_valid else "NO"
        lines = [
            "=" * 50, REPORT_TITLE, f"Hypothesis : {REPORT_HYPOTHESIS}",
            f"Module     : {REPORT_MODULE}", "=" * 50,
            f"Rows Checked           : {self.rows_checked:,}", "",
            f"Errors                 : {len(self.errors())}",
            f"Warnings               : {len(self.warnings())}", "",
            f"Validation Status      : {status}", "",
            f"Data Safe For Research : {data_safe}", "=" * 50,
        ]
        if self.issues:
            lines.append("")
            for issue in self.issues:
                lines.append(f"[{issue.severity}] {issue.check}")
                lines.append(issue.message)
                lines.append("")
            lines.pop()
        return "\n".join(lines)


def _row_numbers(mask: pd.Series) -> List[int]:
    return [position + 1 for position, keep in enumerate(mask.tolist()) if bool(keep)]


def check_dataframe_not_empty(df):
    if df is None or df.empty:
        return [ValidationIssue("ERROR", "empty_dataframe", "DataFrame is empty or None - no rows to validate.")]
    return []


def check_required_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return [ValidationIssue("ERROR", "missing_columns", f"Missing required column(s): {missing}")]
    return []


def check_missing_values(df):
    issues = []
    for col in REQUIRED_COLUMNS:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            issues.append(ValidationIssue("ERROR", "missing_values", f"Column '{col}' has {missing_count} missing value(s)."))
    return issues


def check_non_numeric_prices(df):
    issues = []
    for col in PRICE_COLUMNS:
        original_na_count = df[col].isna().sum()
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_numeric_count = coerced.isna().sum() - original_na_count
        if non_numeric_count > 0:
            issues.append(ValidationIssue("ERROR", "non_numeric_prices", f"Column '{col}' has {non_numeric_count} non-numeric value(s)."))
    return issues


def check_invalid_dates(df):
    original_na_count = df[TIMESTAMP_COL].isna().sum()
    parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    invalid_count = parsed.isna().sum() - original_na_count
    if invalid_count > 0:
        return [ValidationIssue("ERROR", "invalid_dates", f"Column '{TIMESTAMP_COL}' has {invalid_count} value(s) that are not valid dates.")]
    return []


def check_duplicate_rows(df):
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        return [ValidationIssue("WARNING", "duplicate_rows", f"Found {duplicate_count} fully duplicated row(s).")]
    return []


def check_duplicate_timestamps(df):
    parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    duplicate_count = parsed.dropna().duplicated().sum()
    if duplicate_count > 0:
        return [ValidationIssue("ERROR", "duplicate_timestamps", f"Found {duplicate_count} duplicate timestamp(s).")]
    return []


def check_timestamp_ordering(df):
    parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce").dropna()
    if not parsed.is_monotonic_increasing:
        return [ValidationIssue("WARNING", "timestamp_ordering", "Timestamps are not sorted in increasing order.")]
    return []


def check_high_low_relationship(df):
    high = pd.to_numeric(df[HIGH_COL], errors="coerce")
    low = pd.to_numeric(df[LOW_COL], errors="coerce")
    comparable = high.notna() & low.notna()
    violating_rows = _row_numbers(comparable & (high < low))
    violations = len(violating_rows)
    if violations > 0:
        return [ValidationIssue("ERROR", "high_low_relationship", f"Found {violations} invalid bar(s).\nRows: {violating_rows}")]
    return []


def check_price_sanity(df):
    """
    Check: every price is positive and finite.

    The logic here:
    An exchange rate can never be zero, negative, or infinite - it's
    always a positive real number. check_non_numeric_prices only asks
    "can this become a float?", and -1.10, 0.0, and float('inf') are
    all perfectly valid floats, so none of them are caught anywhere
    else. This check runs independently of high_low_relationship and
    open_close_within_range because a bar can be internally consistent
    (e.g. all four prices negative, or all four infinite) and still be
    nonsense - self-consistency doesn't imply the values are real prices.
    """
    issues = []
    for col in PRICE_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        comparable = values.notna()

        non_positive = comparable & (values <= 0)
        non_positive_count = int(non_positive.sum())
        if non_positive_count > 0:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    check="non_positive_prices",
                    message=(
                        f"Column '{col}' has {non_positive_count} value(s) <= 0.\n"
                        f"Rows: {_row_numbers(non_positive)}"
                    ),
                )
            )

        non_finite = comparable & ~np.isfinite(values)
        non_finite_count = int(non_finite.sum())
        if non_finite_count > 0:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    check="non_finite_prices",
                    message=(
                        f"Column '{col}' has {non_finite_count} infinite value(s).\n"
                        f"Rows: {_row_numbers(non_finite)}"
                    ),
                )
            )
    return issues


def check_timestamp_gaps(df):
    """
    Check: consecutive bars are spaced by the expected bar interval.

    WHY the expected interval is inferred with mode() instead of a
    hardcoded constant: this file is meant to work across timeframes
    (M15/H1/H4/D1/...) without a config change every time it's pointed
    at a different file - the most common gap between consecutive bars
    in a given dataset is, almost by definition, that dataset's bar
    size, so the caller never has to tell us.

    WHY this is a WARNING, not an ERROR:
    every real market data feed has legitimate gaps - weekends, exchange
    holidays, broker maintenance windows. This function has no concept
    of a market calendar, so it can't tell a normal 3-day weekend
    closure apart from a 3-day feed outage - only a human reviewing the
    listed timestamps can. Flagging every anomaly as an ERROR would make
    ordinary, correct FX data fail validation every single Monday.

    Note: if check_timestamp_ordering also fired a warning, treat this
    check's output with caution until the data is sorted - out-of-order
    rows produce meaningless (and sometimes negative) diffs.
    """
    parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce").dropna()
    if len(parsed) < 3:
        return []  # too few rows to infer a reliable interval

    diffs = parsed.diff()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        return []

    expected_interval = positive_diffs.mode().iloc[0]
    gap_mask = diffs > expected_interval
    gap_count = int(gap_mask.sum())
    if gap_count == 0:
        return []

    gap_ends = parsed[gap_mask]
    gap_sizes = diffs[gap_mask]
    largest = gap_sizes.sort_values(ascending=False).head(10)
    lines = [
        f"  {gap_ends.loc[idx]} (gap of {size}, expected {expected_interval})"
        for idx, size in largest.items()
    ]

    return [
        ValidationIssue(
            severity="WARNING",
            check="timestamp_gaps",
            message=(
                f"Found {gap_count} gap(s) larger than the inferred bar "
                f"interval ({expected_interval}). Some of these are likely "
                f"normal (weekends, holidays) - review the timestamps below.\n"
                f"Largest gaps (timestamp is the bar right after the gap):\n"
                + "\n".join(lines)
                + (f"\n  ... and {gap_count - len(lines)} more" if gap_count > len(lines) else "")
            ),
        )
    ]


def check_price_outliers(df, z_threshold=RANGE_OUTLIER_Z_THRESHOLD):
    """
    Check: no single bar's high-low range is a wild statistical outlier
    compared to the rest of the series.

    The logic here:
    A single bad tick (e.g. a misplaced decimal) usually shows up as one
    bar with a wildly disproportionate range, while high_low_relationship
    and open_close_within_range can both still pass, because the bad bar
    is internally self-consistent. We use a robust z-score (median +
    median absolute deviation) instead of mean/std, because a single
    extreme outlier can inflate a mean/std enough to hide itself.

    WHY this is a WARNING, not an ERROR:
    genuine markets do produce rare, extreme-range bars (major news
    events, flash crashes) - see e.g. the ECB/Brexit/CPI bars found in
    the EURUSD files this was tested against. Flagging one as an ERROR
    could stop a legitimate backtest on real data. This check exists to
    point a human at bars worth a second look, not to declare them wrong.
    """
    high = pd.to_numeric(df[HIGH_COL], errors="coerce")
    low = pd.to_numeric(df[LOW_COL], errors="coerce")
    comparable = high.notna() & low.notna()
    if comparable.sum() < 20:
        return []  # not enough bars for a meaningful outlier threshold

    bar_range = high - low
    median = bar_range[comparable].median()
    abs_dev = (bar_range - median).abs()
    mad = abs_dev[comparable].median()
    if mad == 0:
        return []  # no meaningful spread to compare against (e.g. synthetic/constant data)

    # 0.6745 rescales MAD to be comparable to a standard deviation under
    # a normal distribution, so the threshold means roughly the same
    # thing regardless of instrument or timeframe.
    z_scores = 0.6745 * abs_dev / mad
    outlier_mask = comparable & (z_scores > z_threshold)
    violating_rows = _row_numbers(outlier_mask)
    violations = len(violating_rows)

    if violations > 0:
        shown = violating_rows[:20]
        return [
            ValidationIssue(
                severity="WARNING",
                check="price_range_outliers",
                message=(
                    f"Found {violations} bar(s) with a high-low range more than "
                    f"{z_threshold}x the typical deviation from the "
                    f"median range (robust z-score).\nRows: {shown}"
                    + (f" ... and {violations - len(shown)} more" if violations > len(shown) else "")
                ),
            )
        ]
    return []


def check_volume_sanity(df, col=VOLUME_COL, threshold=LOW_VOLUME_THRESHOLD):
    """
    Check: tick_volume, if present, is non-negative; flag suspiciously
    thin bars separately.

    WHY this check is skipped entirely when the column is absent:
    tick_volume is documented as optional - not every symbol/vendor
    provides it, and this file shouldn't force it to exist.

    WHY negative volume is an ERROR but low volume is only a WARNING:
    a negative tick count is physically impossible - it can only come
    from a corrupted field or a bad cast upstream. A very low tick count
    (1-2 ticks) is real and does happen (illiquid session opens, holiday
    thin markets - this was confirmed against the real EURUSD M15 file,
    which has a handful of 2-tick bars right at the Sunday-evening open).
    That's worth a human's attention before it feeds a volume-sensitive
    signal, but it isn't wrong data, so it doesn't belong at ERROR level.
    """
    if col not in df.columns:
        return []

    volume = pd.to_numeric(df[col], errors="coerce")
    comparable = volume.notna()
    issues = []

    negative_mask = comparable & (volume < 0)
    negative_count = int(negative_mask.sum())
    if negative_count > 0:
        issues.append(
            ValidationIssue(
                severity="ERROR",
                check="negative_volume",
                message=(
                    f"Column '{col}' has {negative_count} negative value(s).\n"
                    f"Rows: {_row_numbers(negative_mask)}"
                ),
            )
        )

    low_mask = comparable & (volume >= 0) & (volume <= threshold)
    low_count = int(low_mask.sum())
    if low_count > 0:
        rows = _row_numbers(low_mask)
        shown = rows[:20]
        issues.append(
            ValidationIssue(
                severity="WARNING",
                check="low_volume_bars",
                message=(
                    f"Found {low_count} bar(s) with {col} <= "
                    f"{threshold} (thin-liquidity prints - not "
                    f"wrong, but worth reviewing before use in volume-"
                    f"sensitive signals).\nRows: {shown}"
                    + (f" ... and {low_count - len(shown)} more" if low_count > len(shown) else "")
                ),
            )
        )
    return issues


def check_open_close_within_range(df):

    open_ = pd.to_numeric(df[OPEN_COL], errors="coerce")
    high = pd.to_numeric(df[HIGH_COL], errors="coerce")
    low = pd.to_numeric(df[LOW_COL], errors="coerce")
    close = pd.to_numeric(df[CLOSE_COL], errors="coerce")
    issues = []
    open_comparable = open_.notna() & high.notna() & low.notna()
    open_in_range = open_[open_comparable].between(low[open_comparable], high[open_comparable])
    open_violations = (~open_in_range).sum()
    if open_violations > 0:
        issues.append(ValidationIssue("ERROR", "open_within_range", f"Found {open_violations} bar(s) where open is outside [low, high]."))
    close_comparable = close.notna() & high.notna() & low.notna()
    close_in_range = close[close_comparable].between(low[close_comparable], high[close_comparable])
    close_violations = (~close_in_range).sum()
    if close_violations > 0:
        issues.append(ValidationIssue("ERROR", "close_within_range", f"Found {close_violations} bar(s) where close is outside [low, high]."))
    return issues


def check_price_logic(df):
    """
    Verify that high >= max(open, close) and low <= min(open, close) for each bar.
    
    This is a wrapper that combines high_low_relationship and 
    open_close_within_range checks.
    """
    issues = []
    issues.extend(check_high_low_relationship(df))
    issues.extend(check_open_close_within_range(df))
    return issues


def check_timestamp_order(df, col=TIMESTAMP_COL):
    """
    Verify timestamps are monotonically increasing with no gaps or duplicates.
    
    Returns list of ValidationIssue objects combining:
    - Duplicate timestamp detection
    - Timestamp ordering verification
    - Timestamp gap detection
    """
    issues = []
    issues.extend(check_duplicate_timestamps(df))
    issues.extend(check_timestamp_ordering(df))
    issues.extend(check_timestamp_gaps(df))
    return issues


def validate_dataframe(df):
    """
    Run all validation checks on a DataFrame and return results as tuple.
    
    Args:
        df: pandas DataFrame with OHLC data
        
    Returns:
        Tuple of (is_valid: bool, errors: list[str], warnings: list[str])
        where errors and warnings are lists of formatted message strings.
    """
    report = validate_ohlc(df)
    
    errors = [
        f"[{issue.check}] {issue.message}" 
        for issue in report.errors()
    ]
    warnings = [
        f"[{issue.check}] {issue.message}" 
        for issue in report.warnings()
    ]
    
    return (report.is_valid, errors, warnings)


def validate_ohlc(df):
    report = ValidationReport()
    report.rows_checked = 0 if df is None else len(df)
    empty_issues = check_dataframe_not_empty(df)
    if empty_issues:
        report.issues.extend(empty_issues)
        return report
    column_issues = check_required_columns(df)
    if column_issues:
        report.issues.extend(column_issues)
        return report
    report.issues.extend(check_missing_values(df))
    report.issues.extend(check_non_numeric_prices(df))
    report.issues.extend(check_price_sanity(df))
    report.issues.extend(check_invalid_dates(df))
    report.issues.extend(check_duplicate_rows(df))
    report.issues.extend(check_duplicate_timestamps(df))
    report.issues.extend(check_timestamp_ordering(df))
    report.issues.extend(check_timestamp_gaps(df))
    report.issues.extend(check_high_low_relationship(df))
    report.issues.extend(check_open_close_within_range(df))
    report.issues.extend(check_price_outliers(df))
    report.issues.extend(check_volume_sanity(df))
    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Real usage: python validate.py path/to/your_data.csv
        csv_path = sys.argv[1]
        df = pd.read_csv(csv_path)
        report = validate_ohlc(df)
        print(f"Validating: {csv_path}\n")
        print(report.summary())
        sys.exit(0 if report.is_valid else 1)

    # No path given - fall back to a tiny, deliberately broken sample
    # dataset so you can see the validator catch real problems even
    # without a CSV on hand. This branch is NOT your data.
    print("No file path given - running built-in demo on a fake broken sample.")
    print("To check a real file: python validate.py path/to/your_data.csv\n")

    sample_data = pd.DataFrame(
        {
            "time_utc": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:15:00",
                "2024-01-01 00:15:00",  # duplicate timestamp
                "not_a_date",           # invalid date
            ],
            "open": [2050.0, 2051.0, 2051.0, 2052.0],
            "high": [2055.0, 2049.0, 2049.0, 2060.0],  # row 2: high < low
            "low": [2048.0, 2050.0, 2050.0, 2050.0],
            "close": [2052.0, 2065.0, 2065.0, 2053.0],  # row 2: close outside range
        }
    )

    report = validate_ohlc(sample_data)
    print(report.summary())
