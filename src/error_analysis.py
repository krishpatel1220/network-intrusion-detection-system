from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import NORMAL_LABEL, TARGET_COLUMN
from src.evaluation import EvaluationResult, confusion_pairs, evaluate_classifier
from src.preprocessing import normalize_class_label


@dataclass
class ErrorAnalysis:
    """Everything derived from comparing predictions against ground truth."""

    evaluation: EvaluationResult
    false_positives: pd.DataFrame = field(repr=False)
    false_negatives: pd.DataFrame = field(repr=False)
    top_confusions: pd.DataFrame = field(repr=False)

    @property
    def fp_count(self) -> int:
        return len(self.false_positives)

    @property
    def fn_count(self) -> int:
        return len(self.false_negatives)


def analyze_errors(
    labelled_results: pd.DataFrame,
    class_labels: list[str],
    label_column: str = TARGET_COLUMN,
    prediction_column: str = "Prediction",
    model_name: str = "model",
    top_confusions_n: int = 5,
) -> ErrorAnalysis:
    for col in (label_column, prediction_column):
        if col not in labelled_results.columns:
            raise KeyError(f"Column {col!r} not found in the results table.")
    truth = labelled_results[label_column].astype(str).str.strip()
    preds = labelled_results[prediction_column].astype(str).str.strip()
    truth = truth.map(normalize_class_label)
    preds = preds.map(normalize_class_label)

    known = [c for c in class_labels]
    mapping = {name: idx for idx, name in enumerate(known)}
    unknown = sorted(set(truth.unique()) - set(mapping))
    if unknown:
        raise KeyError(
            "Ground-truth column contains labels the model was not trained "
            f"on: {', '.join(unknown)}"
        )

    y_true = truth.map(mapping)
    y_pred = preds.map(mapping).fillna(mapping.get(NORMAL_LABEL, 0)).astype(int)

    evaluation = evaluate_classifier(
        y_true, y_pred, known, model_name=model_name, split="upload"
    )

    fp_mask = (truth == NORMAL_LABEL) & (preds != NORMAL_LABEL)
    fn_mask = (truth != NORMAL_LABEL) & (preds != truth)

    display_cols = [c for c in ("Record Number", label_column, prediction_column, "Severity", "Confidence") if c in labelled_results.columns]
    false_positives = labelled_results.loc[fp_mask, display_cols].copy()
    false_negatives = labelled_results.loc[fn_mask, display_cols].copy()
    if "Confidence" in false_positives.columns:
        false_positives = false_positives.sort_values("Confidence", ascending=False)
    if "Confidence" in false_negatives.columns:
        false_negatives = false_negatives.sort_values("Confidence", ascending=False)

    return ErrorAnalysis(
        evaluation=evaluation,
        false_positives=false_positives.reset_index(drop=True),
        false_negatives=false_negatives.reset_index(drop=True),
        top_confusions=confusion_pairs(evaluation, top_n=top_confusions_n),
    )
