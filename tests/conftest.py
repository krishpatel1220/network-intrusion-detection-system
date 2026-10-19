from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

N_FEATURES = len(FEATURE_COLUMNS)


def make_flows(
    n_normal: int = 120,
    n_dos: int = 40,
    n_scan: int = 25,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def block(n: int, label: str, port: float, dur: float, size: float) -> pd.DataFrame:
        data = {}
        for i, col in enumerate(FEATURE_COLUMNS):
            scale = abs(port) if "Port" in col else max(1.0, dur * (i + 1))
            data[col] = rng.normal(loc=scale, scale=max(0.1, scale * 0.05), size=n)
        data["Destination Port"] = rng.normal(port, 2, n)
        data["Flow Duration"] = np.abs(rng.normal(dur, dur * 0.05, n))
        data["Packet Length Mean"] = np.abs(rng.normal(size, size * 0.05, n))
        data[TARGET_COLUMN] = label
        return pd.DataFrame(data)

    return pd.concat(
        [
            block(n_normal, "Normal Traffic", port=443, dur=10.0, size=400.0),
            block(n_dos, "DoS", port=80, dur=0.05, size=1400.0),
            block(n_scan, "Port Scanning", port=22, dur=0.01, size=60.0),
        ],
        ignore_index=True,
    )


@pytest.fixture
def flows() -> pd.DataFrame:
    """Clean synthetic flows."""
    return make_flows()


@pytest.fixture
def dirty_flows(flows: pd.DataFrame) -> pd.DataFrame:
    df = flows.copy()
    df.loc[0, "Flow Bytes/s"] = np.inf
    df.loc[1, "Packet Length Std"] = np.nan
    df = pd.concat([df, df.head(5)], ignore_index=True)
    conflict = df.iloc[[10]].copy()
    conflict[TARGET_COLUMN] = "DDoS"
    df = pd.concat([df, conflict], ignore_index=True)
    raw = df.head(2).copy()
    raw[TARGET_COLUMN] = "BENIGN"
    df = pd.concat([df, raw], ignore_index=True)
    excluded = df.head(1).copy()
    excluded[TARGET_COLUMN] = "Heartbleed"
    df = pd.concat([df, excluded], ignore_index=True)
    return df
@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory):
    """A tiny trained ModelBundle used by inference/explainability tests."""
    from src.artifacts import ModelBundle
    from src.preprocessing import clean_flows, split_dataset
    from src.training import train_model, model_specs

    df = make_flows(n_normal=160, n_dos=60, n_scan=40)
    cleaned, _ = clean_flows(df, feature_columns=FEATURE_COLUMNS)
    split = split_dataset(cleaned, FEATURE_COLUMNS, random_state=0)
    spec = next(s for s in model_specs() if s.name == "xgboost")
    trained = train_model(spec, split=split, random_state=0, n_jobs=1)

    bundle = ModelBundle(
        model_name="xgboost",
        model_label="XGBoost",
        estimator=trained.estimator,
        label_encoder=trained.label_encoder,
        feature_names=list(FEATURE_COLUMNS),
        metrics={
            name: {"f1_macro": r.f1_macro}
            for name, r in trained.evaluations.items()
        },
    )
    bundle.validate()
    return bundle
