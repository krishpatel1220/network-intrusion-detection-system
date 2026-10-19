from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    HEALTH_THRESHOLDS,
    NORMAL_LABEL,
    NORMAL_LABEL_ALIASES,
    SEVERITY_RECOMMENDATIONS,
)
from src.artifacts import ModelBundle


class PredictionInputError(ValueError):
    """Raised when input data cannot be aligned to the model schema."""


def align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise PredictionInputError("The input table is empty.")
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise PredictionInputError(
            "The input is missing "
            f"{len(missing)} required traffic feature(s): "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )
    aligned = df[feature_names].copy()
    for col in feature_names:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
    return aligned


@dataclass
class PredictionBatch:
    """Predictions joined back onto the caller's original rows."""

    results: pd.DataFrame
    probabilities: pd.DataFrame | None

    @property
    def labels(self) -> pd.Series:
        return self.results["Prediction"]


def predict_flows(bundle: ModelBundle, df: pd.DataFrame) -> PredictionBatch:
    features = align_features(df, bundle.feature_names)

    encoded = np.asarray(bundle.estimator.predict(features)).astype(int)
    decoded = np.asarray(bundle.label_encoder.inverse_transform(encoded))
    labels = pd.Series(
        [normalize_prediction_label(label) for label in decoded], index=df.index
    )

    confidences: pd.Series | None = None
    proba_frame: pd.DataFrame | None = None
    if hasattr(bundle.estimator, "predict_proba"):
        proba = np.asarray(bundle.estimator.predict_proba(features))
        if proba.shape[1] == len(bundle.classes):
            class_names = [normalize_prediction_label(c) for c in bundle.classes]
            confidences = pd.Series(proba.max(axis=1), index=df.index)
            proba_frame = pd.DataFrame(proba, columns=class_names, index=df.index)

    result = df.copy()
    result.insert(0, "Record Number", np.arange(1, len(result) + 1))
    result.insert(1, "Prediction", labels)
    result.insert(2, "Severity", labels.map(bundle.severity_for))
    if confidences is not None:
        result.insert(3, "Confidence", confidences.round(4))

    return PredictionBatch(results=result, probabilities=proba_frame)


def normalize_prediction_label(label: object) -> str:
    """Map any benign alias to :data:`config.NORMAL_LABEL`."""
    text = str(label).strip()
    return NORMAL_LABEL if text.lower() in NORMAL_LABEL_ALIASES else text


@dataclass
class PredictionSummary:
    """Aggregate verdict for a batch of predictions."""

    total_records: int
    normal_count: int
    attack_count: int
    attack_rate: float
    health: str
    recommendation: str

    def as_dict(self) -> dict:
        return {
            "total_records": self.total_records,
            "normal_count": self.normal_count,
            "attack_count": self.attack_count,
            "attack_rate": self.attack_rate,
            "health": self.health,
            "recommendation": self.recommendation,
        }


def summarize_predictions(result_df: pd.DataFrame) -> PredictionSummary:
    total = len(result_df)
    normal_count = int((result_df["Prediction"] == NORMAL_LABEL).sum())
    attack_count = total - normal_count
    attack_rate = round(attack_count / total * 100, 2) if total else 0.0

    for threshold, health, recommendation in HEALTH_THRESHOLDS:
        if attack_rate < threshold:
            break
    else: 
        health, recommendation = HEALTH_THRESHOLDS[-1][1], HEALTH_THRESHOLDS[-1][2]

    return PredictionSummary(
        total_records=total,
        normal_count=normal_count,
        attack_count=attack_count,
        attack_rate=attack_rate,
        health=health,
        recommendation=recommendation,
    )


def recommendation_for(severity: str) -> str:
    """Analyst guidance for a triage severity."""
    return SEVERITY_RECOMMENDATIONS.get(
        severity, "Review this flow manually to confirm whether it requires action."
    )
