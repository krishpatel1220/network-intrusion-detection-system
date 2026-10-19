from __future__ import annotations
from src.preprocessing import DatasetSplit
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluation import EvaluationResult, evaluate_classifier

DEFAULT_N_JOBS = -1


@dataclass
class ModelSpec:

    name: str
    label: str
    builder: Callable[..., Any]
    scale: bool = False
    use_sample_weight: bool = False
    supports_eval_set: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def build(self, random_state: int, n_jobs: int = DEFAULT_N_JOBS) -> Any:
        """Construct a fresh, untrained estimator (optionally inside a
        scaling pipeline)."""
        import inspect

        kwargs = dict(self.params)
        kwargs.setdefault("random_state", random_state)
        accepted = inspect.signature(self.builder).parameters
        if "n_jobs" in accepted:
            kwargs.setdefault("n_jobs", n_jobs)
        estimator = self.builder(**kwargs)
        if self.scale:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            # Pipeline names keep the classifier accessible as ``clf`` and
            # guarantee the scaler is fit on training rows only.
            return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
        return estimator


def _logistic_regression(random_state: int = 0, **extra: Any) -> Any:
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=random_state, **extra
    )


def _decision_tree(random_state: int = 0, **extra: Any) -> Any:
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(
        class_weight="balanced", random_state=random_state, **extra
    )


def _random_forest(random_state: int = 0, n_jobs: int = -1, **extra: Any) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=100, class_weight="balanced_subsample",
        random_state=random_state, n_jobs=n_jobs, **extra,
    )


def _xgboost(random_state: int = 0, n_jobs: int = -1, **extra: Any) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=400,
        learning_rate=0.1,
        max_depth=8,
        early_stopping_rounds=25,
        n_jobs=n_jobs,
        random_state=random_state,
        **extra,
    )


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logistic_regression",
            label="Logistic Regression",
            builder=_logistic_regression,
            scale=True,
        ),
        ModelSpec(
            name="decision_tree",
            label="Decision Tree",
            builder=_decision_tree,
        ),
        ModelSpec(
            name="random_forest",
            label="Random Forest",
            builder=_random_forest,
        ),
        ModelSpec(
            name="xgboost",
            label="XGBoost",
            builder=_xgboost,
            use_sample_weight=True,
            supports_eval_set=True,
        ),
    ]


@dataclass
class TrainedModel:

    name: str
    label: str
    estimator: Any
    label_encoder: Any = None  # sklearn LabelEncoder fitted on training labels
    evaluations: dict[str, EvaluationResult] = field(default_factory=dict)
    best_iteration: int | None = None
    fit_seconds: float = 0.0

    @property
    def classes(self) -> list[str]:
        """Human-readable class names, in encoder order."""
        return [str(c) for c in self.label_encoder.classes_]


def _resolve_clf(estimator: Any) -> Any:
    return estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator


def train_model(
    spec: ModelSpec,
    split: "DatasetSplit | None" = None,
    *,
    X_train: pd.DataFrame | None = None,
    y_train: pd.Series | None = None,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    random_state: int = 42,
    n_jobs: int = DEFAULT_N_JOBS,
    progress: Callable[[str], None] | None = None,
) -> TrainedModel:
    import time

    if split is not None:
        X_train, y_train = split.X_train, split.y_train
        X_val, y_val = split.X_val, split.y_val
        X_test, y_test = split.X_test, split.y_test
    if X_train is None or y_train is None:
        raise ValueError("Provide either a DatasetSplit or X_train/y_train.")
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y_train_enc = pd.Series(encoder.fit_transform(y_train), index=y_train.index)
    y_val_enc = (
        pd.Series(encoder.transform(y_val), index=y_val.index)
        if y_val is not None and len(y_val) > 0
        else None
    )
    y_test_enc = (
        pd.Series(encoder.transform(y_test), index=y_test.index)
        if y_test is not None and len(y_test) > 0
        else None
    )

    estimator = spec.build(random_state=random_state, n_jobs=n_jobs)

    fit_kwargs: dict[str, Any] = {}
    if spec.use_sample_weight:
        fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train_enc)

    clf = _resolve_clf(estimator)
    if spec.supports_eval_set and y_val_enc is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val_enc)]
        fit_kwargs["verbose"] = False

    if progress:
        progress(f"Training {spec.label} ...")

    start = time.perf_counter()
    estimator.fit(X_train, y_train_enc, **fit_kwargs)
    elapsed = time.perf_counter() - start

    trained = TrainedModel(
        name=spec.name,
        label=spec.label,
        estimator=estimator,
        label_encoder=encoder,
        fit_seconds=elapsed,
    )
    best_iteration = getattr(clf, "best_iteration", None)
    if best_iteration is not None and int(best_iteration) > 0:
        trained.best_iteration = int(best_iteration)

    class_labels = [str(c) for c in encoder.classes_]
    eval_inputs: list[tuple[str, pd.DataFrame, pd.Series]] = []
    if y_val_enc is not None:
        eval_inputs.append(("validation", X_val, y_val_enc))
    if y_test_enc is not None:
        eval_inputs.append(("test", X_test, y_test_enc))

    for split_name, X_part, y_part in eval_inputs:
        preds = estimator.predict(X_part)
        trained.evaluations[split_name] = evaluate_classifier(
            y_part, preds, class_labels, model_name=spec.label, split=split_name
        )

    if progress:
        progress(f"{spec.label} trained in {elapsed:.1f}s")
    return trained


def train_all_models(
    split: "DatasetSplit",
    specs: list[ModelSpec] | None = None,
    random_state: int = 42,
    n_jobs: int = DEFAULT_N_JOBS,
    progress: Callable[[str], None] | None = None,
) -> list[TrainedModel]:
    specs = specs if specs is not None else model_specs()
    return [
        train_model(spec, split=split, random_state=random_state, n_jobs=n_jobs, progress=progress)
        for spec in specs
    ]


def experiment_config(
    split: "DatasetSplit",
    random_state: int,
    dataset_info: dict[str, Any],
    n_jobs: int = DEFAULT_N_JOBS,
) -> dict[str, Any]:
    import platform

    import joblib
    import sklearn
    import xgboost

    specs = model_specs()
    return {
        "random_state": random_state,
        "n_jobs": n_jobs,
        "split_sizes": split.sizes(),
        "train_test_overlap_rows": split.train_test_overlap_rows,
        "dataset": dataset_info,
        "class_distribution": {
            "train": {str(k): int(v) for k, v in split.y_train.value_counts().items()},
            "validation": {str(k): int(v) for k, v in split.y_val.value_counts().items()},
            "test": {str(k): int(v) for k, v in split.y_test.value_counts().items()},
        },
        "models": {spec.name: dict(spec.params) for spec in specs},
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
