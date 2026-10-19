from __future__ import annotations

import json
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from config import (
    FULL_DATASET_PATH,
    SAMPLE_OVERVIEW_PATH,
    SAMPLE_PREVIEW_PATH,
    TARGET_COLUMN,
)

SUPPORTED_EXTENSIONS: tuple[str, ...] = ("csv", "xlsx", "xls", "json")

_CHUNK_ROWS = 50_000


class FlowFileError(ValueError):
    pass


def detect_file_type(filename: str) -> str:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTENSIONS:
        raise FlowFileError(
            f"Unsupported file type '.{ext}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    return ext


def load_flow_file(file: str | Path | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    if isinstance(file, (str, Path)):
        source_name = str(file)
        ext = detect_file_type(source_name)
        opener = file
    else:
        if not filename:
            raise FlowFileError("A file name is required to detect the upload format.")
        source_name = filename
        ext = detect_file_type(filename)
        opener = file

    try:
        if ext == "csv":
            df = pd.read_csv(opener)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(opener)
        else:  # json
            df = pd.read_json(opener)
    except Exception as exc:  # narrow the raw parser errors into a clear message
        raise FlowFileError(
            f"Could not parse '{source_name}' as {ext.upper()}: {exc}"
        ) from exc

    if df.empty:
        raise FlowFileError(f"'{source_name}' contains no rows.")
    if df.shape[1] == 0:
        raise FlowFileError(f"'{source_name}' contains no columns.")
    return df


def load_dataset_summary(
    dataset_path: Path | None = None,
    preview_path: Path | None = None,
    overview_path: Path | None = None,
) -> dict[str, Any]:
    dataset_path = dataset_path or FULL_DATASET_PATH
    preview_path = preview_path or SAMPLE_PREVIEW_PATH
    overview_path = overview_path or SAMPLE_OVERVIEW_PATH

    if not dataset_path.exists():
        if not preview_path.exists() or not overview_path.exists():
            raise FileNotFoundError(
                "Neither the full processed dataset "
                f"({dataset_path.name}) nor the packaged preview/overview "
                "assets are available."
            )
        with overview_path.open(encoding="utf-8") as fh:
            overview = json.load(fh)
        return {
            "rows": overview["rows"],
            "columns": overview["columns"],
            "missing": overview["missing"],
            "preview": pd.read_csv(preview_path),
            "column_info": pd.DataFrame(overview["column_info"]),
            "class_counts": pd.DataFrame(overview["class_counts"]),
            "full_sample": False,
        }

    total_rows = 0
    missing_by_column: pd.Series | None = None
    preview: pd.DataFrame | None = None
    columns: pd.Index | None = None
    label_counts: dict[str, int] = {}

    for chunk in pd.read_csv(dataset_path, chunksize=_CHUNK_ROWS):
        total_rows += len(chunk)
        if preview is None:
            preview = chunk.head(10).copy()
            columns = chunk.columns
            missing_by_column = pd.Series(0, index=chunk.columns, dtype="int64")
        assert missing_by_column is not None
        missing_by_column = missing_by_column.add(chunk.isna().sum(), fill_value=0)
        if TARGET_COLUMN in chunk:
            counts = chunk[TARGET_COLUMN].value_counts()
            for label, count in counts.items():
                label_counts[str(label)] = label_counts.get(str(label), 0) + int(count)

    if preview is None or columns is None or missing_by_column is None:
        raise ValueError(f"The dataset '{dataset_path}' is empty.")

    column_info = pd.DataFrame(
        {
            "Column": columns.astype(str),
            "Data Type": preview.dtypes.astype(str).reindex(columns).values,
            "Missing": missing_by_column.reindex(columns).fillna(0).astype(int).values,
        }
    )
    class_counts = pd.DataFrame(
        sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True),
        columns=[TARGET_COLUMN, "Records"],
    )
    return {
        "rows": total_rows,
        "columns": len(columns),
        "missing": int(missing_by_column.sum()),
        "preview": preview,
        "column_info": column_info,
        "class_counts": class_counts,
        "full_sample": True,
    }
