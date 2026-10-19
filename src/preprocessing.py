from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    EXCLUDED_RAW_LABELS,
    LABEL_MAPPING,
    NORMAL_LABEL,
    NORMAL_LABEL_ALIASES,
    TARGET_COLUMN,
)


@dataclass
class CleaningReport:

    rows_in: int = 0
    rows_out: int = 0
    inf_cells_replaced: int = 0
    nan_rows_dropped: int = 0
    duplicate_rows_dropped: int = 0
    conflicting_rows_dropped: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    excluded_label_counts: dict[str, int] = field(default_factory=dict)
    unmapped_label_counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_dropped(self) -> int:
        return self.rows_in - self.rows_out

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "inf_cells_replaced": self.inf_cells_replaced,
            "nan_rows_dropped": self.nan_rows_dropped,
            "duplicate_rows_dropped": self.duplicate_rows_dropped,
            "conflicting_rows_dropped": self.conflicting_rows_dropped,
            "label_counts": dict(self.label_counts),
            "excluded_label_counts": dict(self.excluded_label_counts),
            "unmapped_label_counts": dict(self.unmapped_label_counts),
        }


def canonicalize_label_text(label: object) -> str:
    text = str(label).strip()
    return text.replace("\ufffd", "\u2013")


def normalize_class_label(label: object) -> str:
    text = canonicalize_label_text(label)
    if text.lower() in NORMAL_LABEL_ALIASES:
        return NORMAL_LABEL
    return LABEL_MAPPING.get(text, text)


def map_raw_labels(labels: pd.Series) -> tuple[pd.Series, dict[str, int], dict[str, int]]:
    from config import CLASS_LABELS

    canonical = labels.map(canonicalize_label_text)

    excluded: dict[str, int] = {}
    unmapped: dict[str, int] = {}

    mapped = canonical.map(LABEL_MAPPING)
    benign_mask = canonical.str.lower().isin(NORMAL_LABEL_ALIASES)
    mapped[benign_mask] = NORMAL_LABEL
    already_canonical = canonical.isin(CLASS_LABELS)
    mapped[already_canonical] = canonical[already_canonical]

    unresolved = mapped.isna()
    if unresolved.any():
        for raw_label, count in canonical[unresolved].value_counts().items():
            key = raw_label.strip()
            if key in EXCLUDED_RAW_LABELS:
                excluded[key] = int(count)
            else:
                unmapped[key] = int(count)
    return mapped, excluded, unmapped


def _replace_infinite(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    numeric = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric.to_numpy(dtype=np.float64, na_value=np.nan))
    count = int(inf_mask.sum())
    if count:
        df = df.replace([np.inf, -np.inf], np.nan)
    return df, count


def _drop_conflicting_duplicates(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, int]:
    if df.empty or not feature_columns:
        return df, 0
    hashes = pd.util.hash_pandas_object(df[feature_columns], index=False)
    labelled = pd.DataFrame({"_hash": hashes.to_numpy(), "_label": df["__label__"].to_numpy()})
    conflicting_hashes = labelled.groupby("_hash")["_label"].nunique()
    conflicting_hashes = set(conflicting_hashes[conflicting_hashes > 1].index)
    if not conflicting_hashes:
        return df, 0
    mask = hashes.isin(set(conflicting_hashes))
    return df.loc[~mask], int(mask.sum())


def clean_flows(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    if target_column not in df.columns:
        raise KeyError(
            f"Dataset must contain a {target_column!r} column; "
            f"found columns: {list(df.columns)[:8]} ..."
        )

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    report = CleaningReport(rows_in=len(work))

    features = list(feature_columns) if feature_columns is not None else [
        c for c in work.columns if c != target_column
    ]
    missing_features = [c for c in features if c not in work.columns]
    if missing_features:
        raise KeyError(
            "Dataset is missing required feature columns: "
            + ", ".join(missing_features[:10])
            + (" ..." if len(missing_features) > 10 else "")
        )

    # --- labels -----------------------------------------------------------
    mapped, excluded, unmapped = map_raw_labels(work[target_column])
    report.excluded_label_counts = excluded
    report.unmapped_label_counts = unmapped
    keep = mapped.notna()
    work = work.loc[keep, features + [target_column]].copy()
    work[target_column] = mapped[keep].astype(str)

    # --- numeric coercion + infinities ------------------------------------
    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work, inf_cells = _replace_infinite(work)
    report.inf_cells_replaced = inf_cells

    # --- NaN rows ----------------------------------------------------------
    nan_rows = int(work[features].isna().any(axis=1).sum())
    if nan_rows:
        work = work.dropna(subset=features)
    report.nan_rows_dropped = nan_rows

    # --- exact duplicates ---------------------------------------------------
    before = len(work)
    work = work.drop_duplicates()
    report.duplicate_rows_dropped = before - len(work)

    # --- conflicting duplicates ---------------------------------------------
    work = work.rename(columns={target_column: "__label__"})
    work, conflicting = _drop_conflicting_duplicates(work, features)
    work = work.rename(columns={"__label__": target_column})
    report.conflicting_rows_dropped = conflicting

    work = work.reset_index(drop=True)
    report.rows_out = len(work)
    report.label_counts = {
        str(k): int(v) for k, v in work[target_column].value_counts().items()
    }
    return work[features + [target_column]], report


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


@dataclass
class DatasetSplit:

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    train_test_overlap_rows: int = 0

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.X_train),
            "validation": len(self.X_val),
            "test": len(self.X_test),
        }


def _feature_hashes(df: pd.DataFrame, feature_columns: list[str]) -> pd.Index:
    return pd.util.hash_pandas_object(df[feature_columns], index=False)


def split_dataset(
    df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    target_column: str = TARGET_COLUMN,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    random_state: int = 42,
) -> DatasetSplit:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")
    if target_column not in df.columns:
        raise KeyError(f"Missing target column {target_column!r}.")

    features = list(feature_columns)
    X = df[features]
    y = df[target_column].astype(str)

    X_train, X_temp, y_train, y_temp = train_test_split_stratified(
        X, y, train_size=train_fraction, random_state=random_state
    )
    if val_fraction > 0:
        relative_val = val_fraction / (1.0 - train_fraction)
        X_val, X_test, y_val, y_test = train_test_split_stratified(
            X_temp, y_temp, train_size=relative_val, random_state=random_state
        )
    else:
        X_val, y_val = X_temp.iloc[:0], y_temp.iloc[:0]
        X_test, y_test = X_temp, y_temp

    train_hashes = set(_feature_hashes(X_train, features))
    test_hashes = _feature_hashes(X_test, features)
    overlap = int(test_hashes.isin(train_hashes).sum())

    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_test_overlap_rows=overlap,
    )


def train_test_split_stratified(
    X: pd.DataFrame, y: pd.Series, train_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Thin wrapper around sklearn's stratified split for DataFrame inputs."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=random_state
    )
