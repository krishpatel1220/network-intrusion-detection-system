from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.artifacts import ModelBundle


def _final_estimator(estimator: Any) -> Any:
    return estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator


def shap_capable(bundle: ModelBundle) -> bool:
    return hasattr(_final_estimator(bundle.estimator), "get_booster")


def _contributions_matrix(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray | None:
    import xgboost as xgb

    estimator = _final_estimator(bundle.estimator)
    if not hasattr(estimator, "get_booster"):
        return None
    dmat = xgb.DMatrix(X, feature_names=list(X.columns), missing=np.nan)
    return np.asarray(estimator.get_booster().predict(dmat, pred_contribs=True))


def _as_class_blocks(matrix: np.ndarray, n_features: int) -> np.ndarray:
    block = n_features + 1
    if matrix.ndim == 3:
        return matrix
    if matrix.ndim == 2 and matrix.shape[1] % block == 0:
        return matrix.reshape(matrix.shape[0], -1, block)
    raise ValueError(
        f"Unexpected contribution shape {matrix.shape} for {n_features} features."
    )


@dataclass
class LocalExplanation:
    predicted_class: str
    base_value: float
    contributions: pd.DataFrame  # columns: Feature, Contribution

    @property
    def total(self) -> float:
        return self.base_value + float(self.contributions["Contribution"].sum())


def explain_prediction(
    bundle: ModelBundle,
    X_row: pd.DataFrame,
    top_n: int = 12,
    predicted_encoded: int | None = None,
) -> LocalExplanation | None:
    if len(X_row) != 1:
        raise ValueError("explain_prediction expects exactly one row.")

    matrix = _contributions_matrix(bundle, X_row)
    if matrix is None:
        return None

    if predicted_encoded is None:
        predicted_encoded = int(np.asarray(bundle.estimator.predict(X_row))[0])

    blocks = _as_class_blocks(matrix, len(bundle.feature_names))
    block = blocks[0, predicted_encoded]  # [bias, contribution per feature]
    base_value = float(block[0])
    contrib = pd.DataFrame(
        {
            "Feature": list(X_row.columns),
            "Contribution": block[1:].astype(float),
        }
    )
    contrib["Abs"] = contrib["Contribution"].abs()
    contrib = contrib.sort_values("Abs", ascending=False).head(top_n).drop(columns="Abs")
    contrib = contrib.reset_index(drop=True)

    predicted_label = str(bundle.label_encoder.inverse_transform([predicted_encoded])[0])
    return LocalExplanation(
        predicted_class=predicted_label,
        base_value=base_value,
        contributions=contrib,
    )


def global_feature_importance(
    bundle: ModelBundle,
    X_sample: pd.DataFrame,
    per_class: bool = False,
) -> pd.DataFrame:
    if len(X_sample) == 0:
        return pd.DataFrame(columns=["Feature", "Importance"])

    matrix = _contributions_matrix(bundle, X_sample)
    n_features = len(bundle.feature_names)

    if matrix is not None:
        blocks = _as_class_blocks(matrix, n_features)  # (rows, classes, block)
        mean_abs = np.abs(blocks[:, :, 1:]).mean(axis=0)  # (classes, features)

        if per_class and mean_abs.shape[0] > 1:
            # One importance row per (class, feature).
            frames = []
            for cls_idx, cls_name in enumerate(bundle.classes):
                frame = pd.DataFrame(
                    {"Feature": list(X_sample.columns), "Importance": mean_abs[cls_idx]}
                )
                frame.insert(0, "Class", cls_name)
                frames.append(frame)
            out = pd.concat(frames, ignore_index=True)
            return out.sort_values(
                ["Class", "Importance"], ascending=[True, False]
            ).reset_index(drop=True)

        # Overall importance: mean |contribution|, averaged over classes.
        frame = pd.DataFrame(
            {"Feature": list(X_sample.columns), "Importance": mean_abs.mean(axis=0)}
        )
        return frame.sort_values("Importance", ascending=False).reset_index(drop=True)

    # Fallback: estimator-native importances.
    estimator = _final_estimator(bundle.estimator)
    if not hasattr(estimator, "feature_importances_"):
        raise TypeError(
            f"{type(estimator).__name__} exposes no feature importances and "
            "is not SHAP-capable."
        )
    frame = pd.DataFrame(
        {
            "Feature": list(X_sample.columns),
            "Importance": np.asarray(estimator.feature_importances_, dtype=float),
        }
    )
    return frame.sort_values("Importance", ascending=False).reset_index(drop=True)
