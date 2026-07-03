"""
validate.py
============

Purpose
-------
Validate raw OHLC(V) market data *before* it is used to calculate any
technical indicators (ATR, EMA, ADX, etc.) or fed into a backtest engine.

Indicators and backtests trust their input completely - they don't check
whether "high" is really the highest price of the bar, or whether a
timestamp is missing. If bad data slips through, you get wrong signals,
and wrong signals produce wrong Sharpe ratios. This module is the single
gate all raw data should pass through first.

Responsibilities
-----------------
This file is ONLY responsible for detecting data quality problems.
It deliberately does NOT:
- fetch data
- clean/fix data (e.g. drop duplicates, forward-fill NaNs)
- calculate indicators
Fixing bad data is a separate, deliberate decision (e.g. "should we
forward-fill this NaN, or drop the row?") and should never happen
silently inside a validator - that would hide the very problems this
file exists to surface.

Inputs
------
A pandas DataFrame with (at minimum) the columns:
    time_utc, open, high, low, close
`tick_volume` may also be present and is not currently validated (see
"Future improvements" below).

Outputs
-------
A `ValidationReport` object: a structured, code-friendly list of issues,
each tagged as an ERROR (data is unsafe to use as-is) or a WARNING (data
is usable but should be reviewed). This file never raises exceptions or
prints anything on its own - it hands the report back and lets the
caller decide what to do (e.g. "abort backtest if not report.is_valid").

Assumptions
-----------
- The DataFrame is already loaded into memory (e.g. from SQLite).
- Column names are lowercase and match the *_COL constants below.
- One row = one OHLC bar for a single symbol (no multi-symbol data
  mixed together in the same frame).

Possible edge cases
--------------------
- Completely empty DataFrame.
- DataFrame missing one or more required columns.
- Prices stored as strings ("1.2345" or "N/A") instead of floats.
- Timestamps that are unparseable strings.
- A bar where high < low (physically impossible, but happens with bad
  data feeds or column mix-ups).
- Duplicate bars from re-running a pipeline without idempotent upserts.

Future improvements
--------------------
- Validate volume (e.g. flag negative volume).
- Make column names configurable (e.g. a small class/config object)
  instead of module-level constants, for symbols with different schemas.
- Add a "suggested fix" per issue (without applying it automatically).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# Column name configuration
# ---------------------------------------------------------------------------
# WHY constants instead of hardcoding "high", "low", etc. everywhere below:
# if your data vendor ever changes column names, you fix it in one place
# instead of hunting through every check function for string literals.
#
# WHY "time_utc" instead of a generic "timestamp":
# this matches the exact column name produced by your MT5/broker export
# for EURUSD (data/raw/*.csv in H001_MACD_BASELINE) - open/high/low/close
# already match generic names, only the time column needed to change.
TIMESTAMP_COL = "time_utc"
OPEN_COL = "open"
HIGH_COL = "high"
LOW_COL = "low"
CLOSE_COL = "close"

REQUIRED_COLUMNS = [TIMESTAMP_COL, OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]
PRICE_COLUMNS = [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]

REPORT_TITLE = "MACD FX Quant Research"
REPORT_HYPOTHESIS = "H001 - MACD Baseline"
REPORT_MODULE = Path(__file__).name


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------
@dataclass
class ValidationIssue:
    """
    A single problem found in the data.

    WHY three separate fields instead of one string message:
    `severity` lets calling code decide programmatically whether to stop
    (e.g. "if not report.is_valid: abort backtest") without parsing
    text. `check` names which rule fired - useful for logging, or for
    unit testing that a specific rule works. `message` is the
    human-readable explanation for a person reading the report.
    """

    severity: str  # "ERROR" (unsafe to use) or "WARNING" (usable, but flagged)
    check: str     # short machine-readable name, e.g. "duplicate_timestamps"
    message: str   # human-readable explanation, includes counts


@dataclass
class ValidationReport:
    """
    The full set of issues found while validating one DataFrame.

    WHY collect *all* issues instead of stopping at the first problem:
    if a dataset has five separate problems, fixing them one at a time
    (re-running validation after each fix) wastes a lot of time. Seeing
    the full list up front is far more useful.
    """

    issues: List[ValidationIssue] = field(default_factory=list)
    rows_checked: int = 0

    @property
    def is_valid(self) -> bool:
        """
        True only if there are zero ERROR-level issues.

        WARNINGs alone don't make data invalid - e.g. "rows are out of
        order" is annoying but trivially fixable with a sort, not a
        sign the data itself is corrupted.
        """
        return not any(issue.severity == "ERROR" for issue in self.issues)

    def errors(self) -> List[ValidationIssue]:
        """Return only the ERROR-level issues."""
        return [i for i in self.issues if i.severity == "ERROR"]

    def warnings(self) -> List[ValidationIssue]:
        """Return only the WARNING-level issues."""
        return [i for i in self.issues if i.severity == "WARNING"]

    def summary(self) -> str:
        """
        Build a clean, human-readable report string.

        WHY this formatting logic lives here instead of in each caller:
        keeps the report format consistent everywhere it's printed or
        logged, and means you only update it in one place later.
        """
        status = "PASSED" if self.is_valid else "FAILED"
        data_safe = "YES" if self.is_valid else "NO"

        lines = [
            "=" * 50,
            REPORT_TITLE,
            f"Hypothesis : {REPORT_HYPOTHESIS}",
            f"Module     : {REPORT_MODULE}",
            "=" * 50,
            f"Rows Checked           : {self.rows_checked:,}",
            "",
            f"Errors                 : {len(self.errors())}",
            f"Warnings               : {len(self.warnings())}",
            "",
            f"Validation Status      : {status}",
            "",
            f"Data Safe For Research : {data_safe}",
            "=" * 50,
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
    """Return 1-based row numbers for rows where mask is True."""
    return [position + 1 for position, keep in enumerate(mask.tolist()) if bool(keep)]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
# WHY each check is its own small function instead of one giant function:
# each rule can be read, tested, and reasoned about on its own. If you
# later want to unit test "does duplicate detection work?", you can call
# check_duplicate_rows() directly without running the whole pipeline.

def check_dataframe_not_empty(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: the DataFrame actually contains rows.

    WHY this check must run first, on its own:
    every check below assumes there is data to look at. Running column
    or duplicate checks on an empty DataFrame either crashes or silently
    reports "0 problems found" - which is misleading, since "no data"
    is itself the problem.
    """
    if df is None or df.empty:
        return [
            ValidationIssue(
                severity="ERROR",
                check="empty_dataframe",
                message="DataFrame is empty or None - no rows to validate.",
            )
        ]
    return []


def check_required_columns(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: all required OHLC columns are present.

    WHY this runs before any content check:
    if "high" doesn't exist as a column, code like df["high"] >= df["low"]
    raises a KeyError. Detecting missing columns explicitly, instead of
    letting later checks crash, lets the caller stop early with a clear
    message instead of a confusing traceback.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return [
            ValidationIssue(
                severity="ERROR",
                check="missing_columns",
                message=f"Missing required column(s): {missing}",
            )
        ]
    return []


def check_missing_values(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: no NaN/None values in the required columns.

    WHY this is an ERROR, not a WARNING:
    a single missing 'close' makes it impossible to compute a return for
    that bar, and a NaN silently propagates into every indicator that
    touches it (e.g. an EMA becomes NaN forever after one gap).
    """
    issues = []
    for col in REQUIRED_COLUMNS:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    check="missing_values",
                    message=f"Column '{col}' has {missing_count} missing value(s).",
                )
            )
    return issues


def check_non_numeric_prices(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: open/high/low/close can all be interpreted as numbers.

    WHY pd.to_numeric(errors="coerce") is used to detect this:
    a column can have dtype 'object' even if most values look numeric -
    e.g. one bad row containing "N/A". Coercing to numeric turns anything
    unparseable into NaN. Comparing the coerced NaN count to the
    *original* NaN count tells us exactly how many *new* NaNs were
    introduced - i.e. how many values were genuinely non-numeric, as
    opposed to already-missing (which check_missing_values reports
    separately, so we don't double-count).
    """
    issues = []
    for col in PRICE_COLUMNS:
        original_na_count = df[col].isna().sum()
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_numeric_count = coerced.isna().sum() - original_na_count
        if non_numeric_count > 0:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    check="non_numeric_prices",
                    message=f"Column '{col}' has {non_numeric_count} non-numeric value(s).",
                )
            )
    return issues


def check_invalid_dates(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: every timestamp can be parsed into an actual date/time.

    WHY this matters even though the column exists:
    a string like "2024-13-45" (invalid month/day) or "not_a_date" won't
    raise an error while stored as TEXT in SQLite - it only fails once
    you try to use it as a real date, e.g. resampling to hourly bars.
    Catching it here means the failure happens right next to the raw
    data, not deep inside some unrelated resampling function later.
    """
    original_na_count = df[TIMESTAMP_COL].isna().sum()
    parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    invalid_count = parsed.isna().sum() - original_na_count
    if invalid_count > 0:
        return [
            ValidationIssue(
                severity="ERROR",
                check="invalid_dates",
                message=(
                    f"Column '{TIMESTAMP_COL}' has {invalid_count} "
                    "value(s) that are not valid dates."
                ),
            )
        ]
    return []


def check_duplicate_rows(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: no two rows are 100% identical across every column.

    WHY this is a WARNING, not an ERROR:
    a fully duplicated row (same timestamp AND same OHLCV values) usually
    comes from re-running a fetch without deduplication. It's fixable
    with a single .drop_duplicates() call and doesn't mean any individual
    data point is wrong - unlike, say, high < low.
    """
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        return [
            ValidationIssue(
                severity="WARNING",
                check="duplicate_rows",
                message=f"Found {duplicate_count} fully duplicated row(s).",
            )
        ]
    return []


def check_duplicate_timestamps(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: no timestamp appears more than once.

    WHY this is separate from check_duplicate_rows, and why it's an
    ERROR: two rows can share a timestamp but have *different* prices
    (e.g. a vendor corrected a bar and appended the fix instead of
    overwriting it). A whole-row duplicate check would miss this
    entirely - and it's arguably worse than a harmless duplicate,
    because "what was the price at time T?" no longer has one answer,
    which silently corrupts any time-indexed join or resample.
    """
    # Parse first so two differently-formatted strings that represent the
    # same instant are correctly treated as duplicates, not as different
    # text.
    parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    duplicate_count = parsed.dropna().duplicated().sum()
    if duplicate_count > 0:
        return [
            ValidationIssue(
                severity="ERROR",
                check="duplicate_timestamps",
                message=f"Found {duplicate_count} duplicate timestamp(s).",
            )
        ]
    return []


def check_timestamp_ordering(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: timestamps are sorted from oldest to newest.

    WHY this is a WARNING, not an ERROR:
    out-of-order rows don't corrupt the data itself - df.sort_values()
    fixes it completely. It's still worth flagging, though, because
    feeding unsorted data into indicators that assume chronological
    order (e.g. a rolling EMA) produces meaningless results with no
    crash to warn you.
    """
    parsed = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce").dropna()
    if not parsed.is_monotonic_increasing:
        return [
            ValidationIssue(
                severity="WARNING",
                check="timestamp_ordering",
                message="Timestamps are not sorted in increasing order.",
            )
        ]
    return []


def check_high_low_relationship(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: high >= low on every bar.

    The logic here:
    By definition, 'high' is the highest traded price during the bar and
    'low' is the lowest traded price. There is no valid market scenario
    where the highest price is below the lowest price. If this happens,
    the columns were likely swapped, mis-mapped, or corrupted upstream.
    """
    high = pd.to_numeric(df[HIGH_COL], errors="coerce")
    low = pd.to_numeric(df[LOW_COL], errors="coerce")

    # Only compare rows where both values parsed to real numbers - rows
    # with missing/non-numeric prices are already reported by earlier
    # checks, so we skip them here rather than compare against NaN.
    comparable = high.notna() & low.notna()
    violating_rows = _row_numbers(comparable & (high < low))
    violations = len(violating_rows)

    if violations > 0:
        return [
            ValidationIssue(
                severity="ERROR",
                check="high_low_relationship",
                message=(
                    f"Found {violations} invalid bar(s).\n"
                    f"Rows: {violating_rows}"
                ),
            )
        ]
    return []


def check_open_close_within_range(df: pd.DataFrame) -> List[ValidationIssue]:
    """
    Check: open and close both fall within [low, high] on every bar.

    The logic here:
    'low' and 'high' are the minimum and maximum traded prices during the
    bar. 'open' and 'close' are themselves traded prices within that same
    bar, so they must satisfy:
        low <= open  <= high
        low <= close <= high
    A violation means the bar's four prices are internally inconsistent -
    for example, a 'close' recorded from a different time window than
    its 'high'/'low'.
    """
    open_ = pd.to_numeric(df[OPEN_COL], errors="coerce")
    high = pd.to_numeric(df[HIGH_COL], errors="coerce")
    low = pd.to_numeric(df[LOW_COL], errors="coerce")
    close = pd.to_numeric(df[CLOSE_COL], errors="coerce")

    issues = []

    open_comparable = open_.notna() & high.notna() & low.notna()
    open_in_range = open_[open_comparable].between(
        low[open_comparable], high[open_comparable]
    )
    open_violations = (~open_in_range).sum()
    if open_violations > 0:
        issues.append(
            ValidationIssue(
                severity="ERROR",
                check="open_within_range",
                message=f"Found {open_violations} bar(s) where open is outside [low, high].",
            )
        )

    close_comparable = close.notna() & high.notna() & low.notna()
    close_in_range = close[close_comparable].between(
        low[close_comparable], high[close_comparable]
    )
    close_violations = (~close_in_range).sum()
    if close_violations > 0:
        issues.append(
            ValidationIssue(
                severity="ERROR",
                check="close_within_range",
                message=f"Found {close_violations} bar(s) where close is outside [low, high].",
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def validate_ohlc(df: pd.DataFrame) -> ValidationReport:
    """
    Run all validation checks on `df` and return one combined report.

    WHY the checks run in this specific order:
    check_dataframe_not_empty and check_required_columns run first, and
    if either finds a problem, we return immediately - every check after
    this point assumes the DataFrame has rows and has the columns it
    needs. Running them on a malformed DataFrame would crash instead of
    producing a useful report.

    Every check after that is independent and safe to run regardless of
    what earlier checks found (each one defends itself with
    pd.to_numeric/.notna() as needed), so we simply run all of them and
    collect every issue they find.
    """
    report = ValidationReport()
    report.rows_checked = 0 if df is None else len(df)

    empty_issues = check_dataframe_not_empty(df)
    if empty_issues:
        report.issues.extend(empty_issues)
        return report  # nothing else can be checked on an empty frame

    column_issues = check_required_columns(df)
    if column_issues:
        report.issues.extend(column_issues)
        return report  # nothing else can be checked on missing columns

    # From here on, df is guaranteed non-empty with all required columns.
    report.issues.extend(check_missing_values(df))
    report.issues.extend(check_non_numeric_prices(df))
    report.issues.extend(check_invalid_dates(df))
    report.issues.extend(check_duplicate_rows(df))
    report.issues.extend(check_duplicate_timestamps(df))
    report.issues.extend(check_timestamp_ordering(df))
    report.issues.extend(check_high_low_relationship(df))
    report.issues.extend(check_open_close_within_range(df))

    return report


if __name__ == "__main__":
    # A tiny, deliberately broken sample dataset so you can see the
    # validator catch real problems the first time you run this file.
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