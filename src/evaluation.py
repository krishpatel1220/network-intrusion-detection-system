from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:

    model_name: str
    split: str
    class_labels: list[str]
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    attack_fpr: float
    per_class: pd.DataFrame = field(repr=False)
    confusion: pd.DataFrame = field(repr=False)
    report_text: str = field(repr=False, default="")

    def summary_row(self) -> dict[str, float | str]:
        return {
            "Model": self.model_name,
            "Split": self.split,
            "Accuracy": self.accuracy,
            "Precision (macro)": self.precision_macro,
            "Recall (macro)": self.recall_macro,
            "F1 (macro)": self.f1_macro,
            "F1 (weighted)": self.f1_weighted,
            "Attack FPR": self.attack_fpr,
        }


def evaluate_classifier(
    y_true: "pd.Series | np.ndarray",
    y_pred: "pd.Series | np.ndarray",
    class_labels: list[str],
    model_name: str = "model",
    split: str = "test",
    normal_label: str = "Normal Traffic",
) -> EvaluationResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not class_labels:
        raise ValueError("class_labels must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must have equal length."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty prediction set.")

    labels = list(range(len(class_labels)))

    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))

    # --- binary attack-vs-benign view ---------------------------------------
    normal_idx = _resolve_normal_index(class_labels, normal_label)
    true_is_attack = y_true != normal_idx
    pred_is_attack = y_pred != normal_idx
    benign = ~true_is_attack
    fp = int((benign & pred_is_attack).sum())
    tn = int((benign & ~pred_is_attack).sum())
    attack_fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    # --- per-class one-vs-rest table -----------------------------------------
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for idx, name in enumerate(class_labels):
        tp = int(matrix[idx, idx])
        fn = int(matrix[idx, :].sum()) - tp
        fp_c = int(matrix[:, idx].sum()) - tp
        support = int(matrix[idx, :].sum())
        denom = fp_c + tn_for_class(matrix, idx)
        rows.append(
            {
                "Class": name,
                "Precision": tp / (tp + fp_c) if (tp + fp_c) else 0.0,
                "Recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "F1": (2 * tp / (2 * tp + fp_c + fn)) if (2 * tp + fp_c + fn) else 0.0,
                "Support": support,
                "False Positive Rate": (fp_c / denom) if denom else 0.0,
            }
        )
    per_class = pd.DataFrame(rows)

    confusion_df = pd.DataFrame(
        matrix,
        index=[f"true: {n}" for n in class_labels],
        columns=[f"pred: {n}" for n in class_labels],
    )

    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=class_labels, zero_division=0
    )

    return EvaluationResult(
        model_name=model_name,
        split=split,
        class_labels=list(class_labels),
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        attack_fpr=float(attack_fpr),
        per_class=per_class,
        confusion=confusion_df,
        report_text=report_text,
    )


def tn_for_class(matrix: np.ndarray, idx: int) -> int:
    """True negatives for class *idx* under one-vs-rest reasoning."""
    total = int(matrix.sum())
    row = int(matrix[idx, :].sum())
    col = int(matrix[:, idx].sum())
    tp = int(matrix[idx, idx])
    return total - row - col + tp


def _resolve_normal_index(class_labels: list[str], normal_label: str) -> int:
    for idx, name in enumerate(class_labels):
        if str(name).strip().lower() == normal_label.strip().lower():
            return idx
    # No benign class present: treat nothing as benign (FPR undefined -> 0).
    return -1


def comparison_table(results: "list[EvaluationResult]") -> pd.DataFrame:
    """Build a flat comparison table from several evaluation results."""
    if not results:
        return pd.DataFrame()
    table = pd.DataFrame([r.summary_row() for r in results])
    return table.sort_values("F1 (macro)", ascending=False).reset_index(drop=True)


def confusion_pairs(result: EvaluationResult, top_n: int = 5) -> pd.DataFrame:
    matrix = result.confusion.to_numpy(dtype=np.int64, copy=True)
    np.fill_diagonal(matrix, 0)
    pairs = []
    for i, true_name in enumerate(result.class_labels):
        for j, pred_name in enumerate(result.class_labels):
            if matrix[i, j] > 0:
                pairs.append(
                    {
                        "True class": true_name,
                        "Predicted as": pred_name,
                        "Count": int(matrix[i, j]),
                    }
                )
    frame = pd.DataFrame(pairs, columns=["True class", "Predicted as", "Count"])
    if frame.empty:
        return frame
    return frame.sort_values("Count", ascending=False).head(top_n).reset_index(drop=True)
