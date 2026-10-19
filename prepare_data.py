from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from config import (
    DATA_DIR,
    DROPPED_FEATURES,
    EXCLUDED_RAW_LABELS,
    FEATURE_COLUMNS,
    FULL_DATASET_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLE_OVERVIEW_PATH,
    SAMPLE_PREVIEW_PATH,
    TARGET_COLUMN,
)
from src.preprocessing import CleaningReport, clean_flows

LABEL_COLUMN_ALIASES = ("label", "attack type", "attack_type", "class")

PREVIEW_ROWS_PER_CLASS = 2


def _find_label_column(columns: list[str]) -> str:
    lowered = {c.strip().lower(): c for c in columns}
    for alias in LABEL_COLUMN_ALIASES:
        if alias in lowered:
            return lowered[alias]
    raise KeyError(
        f"Could not find a label column among: {columns[:5]} ... "
        f"(accepted names: {LABEL_COLUMN_ALIASES})"
    )


def load_raw_dataset(raw_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(raw_dir.glob("*.parquet"))
    csv_files = sorted(raw_dir.glob("*.csv"))
    files = parquet_files or csv_files
    if not files:
        raise FileNotFoundError(
            f"No .parquet or .csv files found in {raw_dir}. Download the "
            "CICIDS2017 MachineLearningCVE files first (see README)."
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        print(f"  reading {path.name} ...", flush=True)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        label_col = _find_label_column(list(df.columns))
        wanted = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(
                f"{path.name} is missing schema columns: {', '.join(missing[:5])}"
            )
        frames.append(
            prereduce(
                df[wanted + [label_col]].rename(columns={label_col: TARGET_COLUMN})
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"  loaded {len(combined):,} raw rows from {len(files)} files")
    return combined


def prereduce(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    features = [c for c in df.columns if c != TARGET_COLUMN]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric = df[features]
    inf_mask = np.isinf(numeric.to_numpy(dtype=np.float64, na_value=np.nan))
    if inf_mask.any():
        df[features] = numeric.mask(
            pd.DataFrame(inf_mask, index=df.index, columns=features)
        )
    return df.dropna(subset=features)


def stratified_preview(df: pd.DataFrame, rows_per_class: int = PREVIEW_ROWS_PER_CLASS) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby(TARGET_COLUMN, sort=True):
        parts.append(group.head(rows_per_class))
    preview = pd.concat(parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return preview
def build_overview(df: pd.DataFrame, preview: pd.DataFrame) -> dict:
    class_counts = df[TARGET_COLUMN].value_counts()
    column_info = [
        {
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Missing": int(df[col].isna().sum()),
        }
        for col in df.columns
    ]
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "class_counts": [
            {"Attack Type": str(k), "Records": int(v)} for k, v in class_counts.items()
        ],
        "column_info": column_info,
    }

def merge_reports(reports: list[CleaningReport]) -> CleaningReport:
    total = CleaningReport()
    for rep in reports:
        total.rows_in += rep.rows_in
        total.rows_out += rep.rows_out
        total.inf_cells_replaced += rep.inf_cells_replaced
        total.nan_rows_dropped += rep.nan_rows_dropped
        total.duplicate_rows_dropped += rep.duplicate_rows_dropped
        total.conflicting_rows_dropped += rep.conflicting_rows_dropped
        for key, counter in (
            ("excluded_label_counts", rep.excluded_label_counts),
            ("unmapped_label_counts", rep.unmapped_label_counts),
        ):
            getattr(total, key).update(
                {
                    k: getattr(total, key).get(k, 0) + v
                    for k, v in counter.items()
                }
            )
    return total


def csv_fingerprint(path: Path) -> dict:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def prepare(
    raw_dir: Path,
    output_path: Path,
    sample_size: int | None = None,
    seed: int = 42,
    preview_per_class: int = PREVIEW_ROWS_PER_CLASS,
) -> dict:
    print(f"Loading raw CICIDS2017 files from {raw_dir} ...")
    raw = load_raw_dataset(raw_dir)

    print("Cleaning (label mapping, numeric coercion, dedup) ...")
    cleaned, report = clean_flows(raw, feature_columns=FEATURE_COLUMNS)
    if report.unmapped_label_counts:
        print(
            "  WARNING: unexpected labels were dropped: "
            f"{report.unmapped_label_counts}",
            file=sys.stderr,
        )
    if len(cleaned) == 0:
        raise SystemExit(
            "Cleaning produced an empty dataset - check the raw files and "
            "label mapping (see warnings above)."
        )
    print(
        f"  {report.rows_in:,} raw rows -> {report.rows_out:,} clean rows "
        f"({report.duplicate_rows_dropped:,} exact duplicates, "
        f"{report.conflicting_rows_dropped:,} conflicting-label rows, "
        f"{report.nan_rows_dropped:,} NaN rows, "
        f"{sum(report.excluded_label_counts.values()):,} excluded-label rows removed)"
    )

    if sample_size and sample_size < len(cleaned):
        print(f"Sampling {sample_size:,} rows (stratified, seed={seed}) ...")
        parts = []
        for label, group in cleaned.groupby(TARGET_COLUMN, sort=False):
            n = max(1, round(sample_size * len(group) / len(cleaned)))
            parts.append(group.sample(n=min(n, len(group)), random_state=seed))
        cleaned = (
            pd.concat(parts)
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        cleaned, _ = clean_flows(cleaned, feature_columns=FEATURE_COLUMNS)
        report.rows_out = len(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"  wrote {output_path} ({len(cleaned):,} rows)")

    preview = stratified_preview(cleaned, rows_per_class=preview_per_class)
    preview.to_csv(SAMPLE_PREVIEW_PATH, index=False)
    overview = build_overview(cleaned, preview)
    with SAMPLE_OVERVIEW_PATH.open("w", encoding="utf-8") as fh:
        json.dump(overview, fh, indent=2)
    print(f"  wrote dashboard fallback: {SAMPLE_PREVIEW_PATH.name}, {SAMPLE_OVERVIEW_PATH.name}")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "CICIDS2017 (UNB CIC), MachineLearningCVE files",
            "raw_dir": str(raw_dir),
            "raw_rows": report.rows_in,
        },
        "schema": {
            "n_features": len(FEATURE_COLUMNS),
            "features": list(FEATURE_COLUMNS),
            "dropped_features": DROPPED_FEATURES,
            "label_mapping": "config.LABEL_MAPPING",
            "excluded_raw_labels": EXCLUDED_RAW_LABELS,
        },
        "cleaning": report.to_dict(),
        "class_counts": {
            str(k): int(v) for k, v in cleaned[TARGET_COLUMN].value_counts().items()
        },
        "dataset_file": csv_fingerprint(output_path),
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote {summary_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=FULL_DATASET_PATH)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size (e.g. 500000 to mirror the "
        "original project's sample). Default: use all cleaned rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    prepare(args.raw_dir, args.output, sample_size=args.sample_size, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
