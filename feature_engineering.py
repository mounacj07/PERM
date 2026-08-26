"""
PERM - Predictive Edge Resource Management
Feature Engineering

Turns raw 1-second telemetry (telemetry_log.csv) into a supervised
training set for the Random Forest forecaster:

  1. Load telemetry_log.csv and phase_log.csv (tolerating schema drift
     across old/new phase_log formats - see _load_phase_log).
  2. Label each telemetry row with its active traffic phase (steady /
     ramp / burst / cooldown) via an as-of join on phase_log timestamps.
     This is kept as metadata only, never as a model feature - the live
     inference service has no way of knowing which Locust phase is
     "currently active", so training on it would leak information the
     real system will never have at prediction time.
  3. Build sliding-window features: each window covers the previous 30
     seconds of telemetry, stepped every 5 seconds - matching the 5s
     cadence inference_service.py will actually query the model at, so
     training examples are generated the same way the model will be used.
  4. Label each window with the actual CPU% and latency observed 30
     seconds after the window ends (joined by timestamp, not row offset,
     so it's robust to any dropped/late telemetry ticks).
  5. Split 70/15/15 along the timeline (not shuffled) to avoid leaking
     future information into training - earliest 70% of windows train,
     next 15% validate, final 15% test. Reports phase distribution per
     split so you can see splits aren't accidentally dominated by one
     traffic profile.

Run:
    python feature_engineering.py
    python feature_engineering.py --telemetry telemetry_log.csv --phases phase_log.csv

Outputs (in the same directory):
    features_train.csv
    features_val.csv
    features_test.csv
"""

import argparse
import csv
import logging
import os

import numpy as np
import pandas as pd

import features as feat_lib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_engineering")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
WINDOW_SEC = 10          # sliding window size: matches inference_service.py WINDOW_SEC
STRIDE_SEC = 3           # step between windows
HORIZON_SEC = 30         # how far ahead we're forecasting
LABEL_TOLERANCE_SEC = 2  # max allowed gap when matching a window to its future label row

# Longest single stage in STAGES (locustfile.py) is 180s (steady/ramp). A
# telemetry row more than this many seconds past its most recent phase_log
# entry almost certainly isn't part of that traffic phase anymore - it's
# either idle background collection (collector left running with no active
# Locust run) or a completely separate, unrelated session. Backward as-of
# joins have no natural cutoff, so this bounds it explicitly.
MAX_PHASE_AGE_SEC = 300

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder (0.15)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_telemetry(path: str) -> pd.DataFrame:
    # encoding="utf-8-sig" strips a UTF-8 byte-order-mark (BOM) if present.
    # Some Windows tools/sync clients (Excel, OneDrive, certain editors)
    # silently prepend a BOM to text files, which otherwise attaches itself
    # to the first column name (turning "timestamp" into "\ufefftimestamp")
    # and causes a confusing KeyError on a column that visibly exists.
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _load_phase_log(path: str) -> pd.DataFrame:
    """
    Tolerant phase_log.csv loader.

    Older versions of locustfile.py wrote a 4-column format
    (timestamp, phase, target_users, elapsed_sec); newer versions added a
    `cycle` column, giving 5 columns. If a phase_log.csv file already
    existed on disk when a run started, the header never gets rewritten
    (only written when the file is first created), so a single real file
    can end up with a stale 4-column header followed by a mix of 4- and
    5-column data rows. A plain pd.read_csv() hard-crashes on that.

    This parses line-by-line and infers the schema per row from its
    field count, so both formats load cleanly out of the same file.
    Rows that don't match either known shape are skipped with a warning
    rather than aborting the whole pipeline.
    """
    records = []
    skipped = 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # discard - we infer per-row instead
        for row in reader:
            if len(row) == 4:
                timestamp, phase, target_users, elapsed_sec = row
                cycle = None
            elif len(row) == 5:
                timestamp, phase, cycle, target_users, elapsed_sec = row
                cycle = int(cycle)
            else:
                skipped += 1
                continue
            records.append({
                "timestamp": timestamp,
                "phase": phase,
                "cycle": cycle,
                "target_users": int(target_users),
                "elapsed_sec": float(elapsed_sec),
            })

    if skipped:
        log.warning("Skipped %d malformed phase_log row(s).", skipped)

    df = pd.DataFrame.from_records(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def label_phases(telemetry: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the active traffic phase to each telemetry row via an as-of
    (backward) join: a telemetry row's phase is whichever phase_log entry
    most recently preceded it in time.

    Two kinds of rows get dropped:
      1. Rows that predate every phase_log entry (no match at all).
      2. Rows more than MAX_PHASE_AGE_SEC past their matched phase_log
         entry. A plain backward as-of join has no upper bound on
         "recently" - without this cutoff, telemetry collected long after
         a run ended (e.g. the collector left running idle, or in this
         project's case, background telemetry collected while chatting
         about this very pipeline) would silently inherit the last real
         phase's label (typically "cooldown") and pollute training with
         windows that have nothing to do with real Locust traffic.
    """
    phase_times = phases[["timestamp", "phase", "cycle"]].rename(
        columns={"timestamp": "phase_timestamp"}
    )
    merged = pd.merge_asof(
        telemetry, phase_times, left_on="timestamp", right_on="phase_timestamp",
        direction="backward",
    )

    no_match_mask = merged["phase"].isna()
    dropped_no_match = int(no_match_mask.sum())
    merged = merged[~no_match_mask].copy()

    merged["phase_age_sec"] = (merged["timestamp"] - merged["phase_timestamp"]).dt.total_seconds()
    stale_mask = merged["phase_age_sec"] > MAX_PHASE_AGE_SEC
    dropped_stale = int(stale_mask.sum())
    merged = merged[~stale_mask].drop(columns=["phase_timestamp", "phase_age_sec"]).reset_index(drop=True)

    if dropped_no_match:
        log.warning(
            "Dropped %d telemetry row(s) with no matching phase_log entry "
            "(predate the earliest known traffic run).",
            dropped_no_match,
        )
    if dropped_stale:
        log.warning(
            "Dropped %d telemetry row(s) more than %ds past their last known "
            "phase transition (idle/background telemetry, not real traffic).",
            dropped_stale, MAX_PHASE_AGE_SEC,
        )

    return merged


# --------------------------------------------------------------------------
# Sliding window feature construction
# --------------------------------------------------------------------------
def _window_features(window_df: pd.DataFrame) -> dict:
    """Delegate to the canonical shared feature extraction in features.py."""
    return feat_lib.window_features(window_df)


def build_feature_table(telemetry: pd.DataFrame) -> pd.DataFrame:
    """
    Slide a WINDOW_SEC-wide window across the telemetry timeline, stepping
    every STRIDE_SEC seconds. For each window, compute summary features and
    look up the actual CPU%/latency observed HORIZON_SEC after the window
    ends to use as the training label.
    """
    telemetry = telemetry.sort_values("timestamp").reset_index(drop=True)
    ts_array = telemetry["timestamp"].values
    ts_ns = ts_array.astype("datetime64[ns]").astype(np.int64)

    start_time = pd.Timestamp(ts_array[0])
    end_time = pd.Timestamp(ts_array[-1])

    rows = []
    window_end = start_time + pd.Timedelta(seconds=WINDOW_SEC)

    while window_end + pd.Timedelta(seconds=HORIZON_SEC) <= end_time:
        window_start = window_end - pd.Timedelta(seconds=WINDOW_SEC)

        ws_ns = window_start.value
        we_ns = window_end.value

        i_start = np.searchsorted(ts_ns, ws_ns, side="left")
        i_end = np.searchsorted(ts_ns, we_ns, side="right")

        window_df = telemetry.iloc[i_start:i_end]

        if len(window_df) < 5:
            window_end += pd.Timedelta(seconds=STRIDE_SEC)
            continue

        features = _window_features(window_df)
        features["window_end"] = window_end
        features["phase"] = window_df["phase"].iloc[-1]

        target_ns = (window_end + pd.Timedelta(seconds=HORIZON_SEC)).value
        target_idx = np.searchsorted(ts_ns, target_ns)
        if target_idx >= len(ts_ns):
            target_idx = len(ts_ns) - 1

        diff_sec = abs((ts_ns[target_idx] - target_ns) / 1e9)
        if target_idx > 0:
            diff_sec_prev = abs((ts_ns[target_idx - 1] - target_ns) / 1e9)
            if diff_sec_prev < diff_sec:
                target_idx = target_idx - 1
                diff_sec = diff_sec_prev

        if diff_sec <= LABEL_TOLERANCE_SEC:
            label_row = telemetry.iloc[target_idx]
            features["target_cpu_percent"] = label_row["cpu_percent"]
            features["target_latency_ms"] = label_row["avg_latency_ms"]
            rows.append(features)

        window_end += pd.Timedelta(seconds=STRIDE_SEC)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Temporal 70/15/15 split
# --------------------------------------------------------------------------
def temporal_split(features: pd.DataFrame):
    """
    Splits along the timeline (earliest -> latest), NOT shuffled. Shuffling
    a time-series split lets the model train on windows that occur after
    windows in the test set - i.e. training on the future to predict the
    past - which silently inflates validation/test scores and doesn't
    reflect how the model will actually be used (always predicting forward
    from the most recent data it's ever seen).
    """
    features = features.sort_values("window_end").reset_index(drop=True)
    n = len(features)
    train_end = int(n * TRAIN_FRAC)
    val_end = train_end + int(n * VAL_FRAC)

    train = features.iloc[:train_end]
    val = features.iloc[train_end:val_end]
    test = features.iloc[val_end:]
    return train, val, test


def _print_phase_distribution(name: str, split_df: pd.DataFrame):
    counts = split_df["phase"].value_counts(normalize=True).round(3) * 100
    parts = ", ".join(f"{phase}={pct:.1f}%" for phase, pct in counts.items())
    print(f"  {name:<6} n={len(split_df):<5} {parts}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PERM feature engineering pipeline")
    parser.add_argument("--telemetry", default="telemetry_log.csv")
    parser.add_argument("--phases", default="phase_log.csv")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    log.info("Loading telemetry from %s ...", args.telemetry)
    telemetry = load_telemetry(args.telemetry)
    log.info("  %d raw telemetry rows, %s -> %s",
             len(telemetry), telemetry['timestamp'].min(), telemetry['timestamp'].max())

    log.info("Loading phase log from %s ...", args.phases)
    phases = _load_phase_log(args.phases)
    log.info("  %d phase transitions loaded", len(phases))

    labeled = label_phases(telemetry, phases)
    log.info("  %d telemetry rows retained after phase labeling", len(labeled))

    log.info(
        "Building sliding-window features (window=%ds, stride=%ds, horizon=%ds) ...",
        WINDOW_SEC, STRIDE_SEC, HORIZON_SEC,
    )
    features = build_feature_table(labeled)
    log.info("  %d feature rows built", len(features))

    if len(features) == 0:
        raise SystemExit(
            "No feature rows were built. This usually means telemetry_log.csv "
            "and phase_log.csv don't overlap in time, or the telemetry span is "
            "shorter than window + horizon (60s minimum)."
        )

    train, val, test = temporal_split(features)

    log.info("Phase distribution per split (diagnostic only — phase is not a model feature):")
    _print_phase_distribution("train", train)
    _print_phase_distribution("val", val)
    _print_phase_distribution("test", test)

    feature_cols = [c for c in features.columns if c not in ("window_end", "phase")]
    meta_cols = ["window_end", "phase"]
    output_cols = meta_cols + [c for c in feature_cols if c not in meta_cols]

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        out_path = os.path.join(args.outdir, f"features_{name}.csv")
        split_df[output_cols].to_csv(out_path, index=False)
        log.info("Wrote %s (%d rows)", out_path, len(split_df))


if __name__ == "__main__":
    main()