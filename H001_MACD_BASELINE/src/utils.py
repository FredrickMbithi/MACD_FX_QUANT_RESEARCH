"""
utils.py
========

Purpose
-------
Hold small, reusable helper functions that main.py (and potentially
other future scripts) needs for plumbing - not for strategy logic.

Responsibilities
-----------------
This file is ONLY responsible for generic infrastructure: reading
files, creating directories, and logging. It deliberately does NOT:
- know what an exit rule, MACD, or crossover is (that's
  execution.py/indicators.py/signals.py)
- validate data quality (that's validate.py)
- decide the workflow order (that's main.py)

WHY this separation matters
-------------------------------
Every helper here would make sense in a completely different project
with a completely different strategy - reading a YAML file, creating a
folder, and writing a log line have nothing to do with MACD or FX. If
a function in this file needs to know what "opposite_signal_exit" means
to work correctly, it belongs in a different file, not here.

Inputs
------
Varies per function - see each docstring below.

Outputs
-------
Varies per function - see each docstring below.

Assumptions
-----------
- config.yaml is valid YAML (this file does not validate its
  structure or required keys - see "Future improvements").
- Callers are responsible for deciding *what* to log and *when* to
  create directories; this file only provides the mechanism.

Possible edge cases
--------------------
- A config file that doesn't exist: load_config() lets the underlying
  FileNotFoundError propagate rather than hiding it, since silently
  returning an empty config would let main.py run with defaults nobody
  chose.
- A directory that already exists: ensure_directory_exists() treats
  this as success, not an error - "make sure this exists" should not
  fail just because it already does.
- Calling setup_logger() more than once with the same name (e.g. if
  main.py is re-run in the same process, as can happen in a notebook):
  handled by clearing any handlers already attached, so log lines don't
  get duplicated on every additional call.

Future improvements
--------------------
- Add schema validation for config.yaml (required keys, expected
  types) - today, a typo'd or missing key in the config only surfaces
  as an error wherever main.py tries to use it, not at load time.
- Support environment-variable overrides for config values (useful for
  running the same config across multiple machines/experiments).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """
    Read a YAML config file into a plain Python dict.

    WHY this belongs in utils.py, not main.py
    ----------------------------------------------
    Turning YAML text into a dict is a generic file-reading operation -
    this function has no idea what any of the keys inside config.yaml
    mean (exit rule, lot_size, data paths, ...). That
    knowledge stays in main.py, which reads specific keys out of the
    dict this function returns.

    Parameters
    ----------
    config_path : str
        Path to a .yaml config file.

    Returns
    -------
    dict
        The parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If `config_path` doesn't exist - this is allowed to propagate
        rather than being caught here, since main.py cannot safely
        continue without its config.
    """
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_ohlc_csv(csv_path: str) -> pd.DataFrame:
    """
    Read a raw OHLC CSV file into a DataFrame.

    WHY this belongs in utils.py, not validate.py
    ---------------------------------------------------
    Reading bytes off disk into a DataFrame has nothing to do with
    whether the data turns out to be valid - that judgment is entirely
    validate.py's job, and it happens afterward, on whatever this
    function returns. Keeping "get the data into memory" and "is the
    data any good" as separate steps means either one can be swapped
    out (e.g. reading from a database instead of a CSV) without
    touching the other.

    WHY this function does NOT parse the timestamp column into a real
    datetime type
    -----------------------------------------------------------------
    Every downstream file (execution.py, export.py) is written to
    assume time_utc stays exactly as it appears in the source CSV
    (e.g. "2016-01-03 22:00:00") - see export.py's format_timestamp()
    for why preserving that exact string matters. Parsing dates here
    would risk pandas silently reformatting them somewhere along the
    pipeline.

    Parameters
    ----------
    csv_path : str
        Path to a raw OHLC CSV file (columns: time_utc, open, high,
        low, close, tick_volume).

    Returns
    -------
    pd.DataFrame
        The raw file contents, completely unvalidated and unmodified -
        pass this straight to validate.validate_ohlc() next.
    """
    return pd.read_csv(csv_path)


def ensure_directory_exists(directory_path: str) -> None:
    """
    Create a directory (and any missing parent directories) if it
    doesn't already exist.

    WHY this belongs in utils.py
    ---------------------------------
    "Make sure this folder exists before I write to it" has nothing to
    do with what gets written there - main.py needs this before saving
    to output/ or output/logs/, but the helper itself doesn't know or
    care whether a trade log or a log file ends up inside.

    Parameters
    ----------
    directory_path : str
        The directory to create if missing.
    """
    # WHY exist_ok=True instead of checking os.path.exists() first:
    # checking-then-creating is a classic race condition (another
    # process could create the directory between the check and the
    # create) - exist_ok=True asks the operating system to do the
    # check-and-create atomically, and simply treats "it already
    # exists" as success rather than an error.
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def generate_run_timestamp() -> str:
    """
    Return the current time as a filename-safe string, e.g.
    "20260704_153000".

    WHY this belongs in utils.py
    ---------------------------------
    Tagging a log file or output CSV with "when was this run" is a
    generic naming convenience - it's useful whether the run is a MACD
    backtest, a data-validation pass, or something unrelated entirely.

    WHY this specific format
    -----------------------------
    Colons (as in a normal time string like "15:30:00") aren't valid in
    filenames on Windows. Using underscores instead
    ("%Y%m%d_%H%M%S") keeps the same information but is always safe to
    use directly inside a filename, on any operating system.

    Returns
    -------
    str
        The current local time, formatted as YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(name: str, log_dir: str, log_filename: str | None = None) -> logging.Logger:
    """
    Create (or return) a logger that writes to both the console and a
    log file.

    WHY logging belongs in utils.py, not scattered as print() calls
    throughout every file
    ---------------------------------------------------------------------
    A print() statement always goes to the console and nowhere else,
    and it can't be turned off or leveled ("this is just informational"
    vs "this is a real problem") without editing the call site. A
    logger centralizes that decision once, here, so every other file
    in this project can log a message without knowing or caring where
    it ends up.

    WHY the existing handlers are cleared before adding new ones
    ------------------------------------------------------------------
    If setup_logger() is called more than once with the same `name`
    (e.g. main.py is re-run in the same Python process, which happens
    often in a notebook or interactive session), Python's logging
    module would otherwise keep attaching a new file/console handler
    on top of the old ones - and every subsequent log message would be
    printed multiple times, once per accumulated handler.

    Parameters
    ----------
    name : str
        Logger name - typically __name__ of the calling module, or a
        fixed string like "H001_MACD_BASELINE".
    log_dir : str
        Directory the log file should be written into. Created if it
        doesn't exist (see ensure_directory_exists()).
    log_filename : str or None, default None
        Name of the log file. If None, one is generated using
        generate_run_timestamp(), so each run gets its own log file
        instead of overwriting the previous run's.

    Returns
    -------
    logging.Logger
        A logger configured with both a console handler and a file
        handler, ready to use (logger.info(...), logger.warning(...),
        etc.).
    """
    ensure_directory_exists(log_dir)

    if log_filename is None:
        log_filename = f"run_{generate_run_timestamp()}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # See "WHY the existing handlers are cleared" above.
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(Path(log_dir) / log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # A small end-to-end demo of every helper in this file, using a
    # temporary config and directory so running this script doesn't
    # touch your real project folders.
    demo_config_path = "demo_config.yaml"
    with open(demo_config_path, "w", encoding="utf-8") as f:
        f.write(
            "symbol: EURUSD\n"
            "exit_rule: opposite_signal\n"
            "lot_size: 1.0\n"
        )

    config = load_config(demo_config_path)
    print("Loaded config:", config)

    ensure_directory_exists("demo_output/logs")
    print("Run timestamp:", generate_run_timestamp())

    logger = setup_logger("utils_demo", "demo_output/logs")
    logger.info("utils.py demo started")
    logger.warning("this is what a warning-level message looks like")
