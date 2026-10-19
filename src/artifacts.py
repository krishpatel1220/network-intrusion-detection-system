from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import CLASS_SEVERITY, DEFAULT_SEVERITY, TARGET_COLUMN


class ArtifactError(RuntimeError):
    """Raised when saved artifacts are missing or mutually inconsistent."""


@dataclass
class ModelBundle:

    model_name: str
    model_label: str
    estimator: Any
    label_encoder: LabelEncoder
    feature_names: list[str]
    class_severity: dict[str, str] = field(default_factory=lambda: dict(CLASS_SEVERITY))
    training_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)
    created_at: str = ""

    # -- derived properties -------------------------------------------------

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.label_encoder.classes_]

    @property
    def normal_index(self) -> int | None:
        for idx, name in enumerate(self.classes):
            if name.strip().lower() == "normal traffic":
                return idx
        return None

    def severity_for(self, class_name: str) -> str:
        return self.class_severity.get(str(class_name).strip().lower(), DEFAULT_SEVERITY)

    # -- validation -----------------------------------------------------------

    def collect_problems(self) -> list[str]:
        problems: list[str] = []

        encoder_classes = np.asarray(self.label_encoder.classes_)
        if encoder_classes.size == 0:
            problems.append("The label encoder has no classes.")

        model_classes = np.asarray(getattr(self.estimator, "classes_", []))
        if model_classes.size:
            if model_classes.size != encoder_classes.size:
                problems.append(
                    f"Class-count mismatch: estimator={model_classes.size}, "
                    f"encoder={encoder_classes.size}."
                )
            if not np.array_equal(model_classes, np.arange(model_classes.size)):
                problems.append(
                    "Estimator classes are not contiguous integers 0..N-1; "
                    "predictions cannot be decoded reliably."
                )

        n_features = int(getattr(self.estimator, "n_features_in_", 0) or 0)
        if n_features and n_features != len(self.feature_names):
            problems.append(
                f"Feature-count mismatch: estimator expects {n_features} "
                f"features, bundle declares {len(self.feature_names)}."
            )

        named = getattr(self.estimator, "feature_names_in_", None)
        if named is not None:
            estimator_features = [str(c) for c in named]
            if estimator_features != list(self.feature_names):
                problems.append(
                    "Estimator feature names/order differ from the bundle's "
                    "feature schema."
                )

        if TARGET_COLUMN in self.feature_names:
            problems.append(
                f"The label column {TARGET_COLUMN!r} must not be part of the "
                "feature schema (it would leak the answer at inference time)."
            )
        return problems

    def validate(self) -> None:
        problems = self.collect_problems()
        if problems:
            raise ArtifactError(
                "Invalid model bundle:\n- " + "\n- ".join(problems)
            )

    # -- persistence -----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path, validate: bool = True) -> "ModelBundle":
        """Load a bundle from *path* and re-validate its consistency."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, ModelBundle):
            raise ArtifactError(
                f"'{path.name}' contains a {type(obj).__name__}, not a "
                "ModelBundle. Retrain with train.py to regenerate artifacts."
            )
        if validate:
            obj.validate()
        return obj

    def summary(self) -> dict[str, Any]:
        """JSON-safe summary used by the dashboard and reports."""
        data = asdict(self)
        data.pop("estimator", None)
        data.pop("label_encoder", None)
        data["classes"] = self.classes
        data["n_features"] = len(self.feature_names)
        return data


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    return bundle.save(path)


def load_bundle(path: str | Path, validate: bool = True) -> ModelBundle:
    return ModelBundle.load(path, validate=validate)


def bundle_path_for(models_dir: str | Path, model_name: str, suffix: str = ".bundle.joblib") -> Path:
    return Path(models_dir) / f"{model_name}{suffix}"


# ---------------------------------------------------------------------------
# Legacy artifacts (pre-bundle format)
# ---------------------------------------------------------------------------


def validate_legacy_artifacts(estimator: Any, encoder: Any) -> list[str]:
    feature_names = [str(c) for c in getattr(estimator, "feature_names_in_", [])]
    if not feature_names:
        feature_names = [
            f"f{i}" for i in range(int(getattr(estimator, "n_features_in_", 0) or 0))
        ]
    bundle = ModelBundle(
        model_name="legacy",
        model_label="Legacy estimator",
        estimator=estimator,
        label_encoder=encoder,
        feature_names=feature_names,
    )
    return bundle.collect_problems()


def load_legacy_artifacts(models_dir: str | Path) -> ModelBundle:
    models_dir = Path(models_dir)
    model_path = models_dir / "network_intrusion_detector.pkl"
    encoder_path = models_dir / "label_encoder.pkl"
    for path in (model_path, encoder_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing legacy artifact: {path}")

    estimator = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    problems = validate_legacy_artifacts(estimator, encoder)
    if problems:
        raise ArtifactError(
            "Legacy artifacts are inconsistent:\n- " + "\n- ".join(problems)
        )

    feature_names = [str(c) for c in getattr(estimator, "feature_names_in_", [])]
    if not feature_names:
        raise ArtifactError(
            "The legacy estimator does not expose feature_names_in_, so the "
            "training feature order cannot be reproduced. Retrain with "
            "train.py."
        )

    return ModelBundle(
        model_name="legacy_xgboost",
        model_label="XGBoost (legacy artifacts)",
        estimator=estimator,
        label_encoder=encoder,
        feature_names=feature_names,
        training_config={"origin": "legacy separate-pickle artifacts"},
    )


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Row count and SHA-256 of a CSV dataset, for experiment tracking."""
    path = Path(path)
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            sha.update(block)
    rows = sum(1 for _ in path.open("rb")) - 1
    return {"path": str(path.name), "rows": rows, "sha256": sha.hexdigest()}
