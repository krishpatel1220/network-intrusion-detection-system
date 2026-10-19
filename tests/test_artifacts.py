
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from config import FEATURE_COLUMNS
from src.artifacts import (
    ArtifactError,
    ModelBundle,
    file_fingerprint,
    load_bundle,
    load_legacy_artifacts,
    save_bundle,
    validate_legacy_artifacts,
)


def _encoder_for(labels: list[str]) -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit(labels)
    return enc


def _tiny_estimator():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(60, len(FEATURE_COLUMNS))), columns=list(FEATURE_COLUMNS)
    )
    y = np.array([0, 1, 2] * 20)
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    return model


@pytest.fixture
def bundle() -> ModelBundle:
    classes = ["DoS", "Normal Traffic", "Port Scanning"]  # encoder order
    return ModelBundle(
        model_name="test",
        model_label="Test model",
        estimator=_tiny_estimator(),
        label_encoder=_encoder_for(classes),
        feature_names=list(FEATURE_COLUMNS),
    )


class TestBundleValidation:
    def test_valid_bundle_passes(self, bundle: ModelBundle):
        bundle.validate()  # must not raise
        assert bundle.collect_problems() == []

    def test_class_count_mismatch_detected(self, bundle: ModelBundle):
        bundle.label_encoder = _encoder_for(["DoS", "Normal Traffic"])
        problems = bundle.collect_problems()
        assert any("Class-count mismatch" in p for p in problems)
        with pytest.raises(ArtifactError, match="Class-count mismatch"):
            bundle.validate()

    def test_feature_count_mismatch_detected(self, bundle: ModelBundle):
        bundle.feature_names = bundle.feature_names[:-1]
        assert any("Feature-count mismatch" in p for p in bundle.collect_problems())

    def test_feature_order_mismatch_detected(self, bundle: ModelBundle):
        bundle.feature_names = list(reversed(bundle.feature_names))
        assert any("names/order differ" in p for p in bundle.collect_problems())

    def test_target_column_in_features_rejected(self, bundle: ModelBundle):
        bundle.feature_names = bundle.feature_names[:-1] + ["Attack Type"]
        assert any("must not be part of the feature schema" in p for p in bundle.collect_problems())


class TestBundlePersistence:
    def test_roundtrip_preserves_metadata(self, bundle: ModelBundle, tmp_path):
        path = save_bundle(bundle, tmp_path / "m.bundle.joblib")
        loaded = load_bundle(path)
        assert loaded.model_name == bundle.model_name
        assert loaded.classes == bundle.classes
        assert loaded.feature_names == bundle.feature_names

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_bundle(tmp_path / "nope.joblib")

    def test_load_non_bundle_pickle_raises(self, tmp_path):
        import joblib

        path = tmp_path / "legacy.pkl"
        joblib.dump({"not": "a bundle"}, path)
        with pytest.raises(ArtifactError, match="not a ModelBundle"):
            load_bundle(path)

    def test_invalid_bundle_refuses_to_save(self, bundle: ModelBundle, tmp_path):
        bundle.label_encoder = _encoder_for(["a", "b"])
        with pytest.raises(ArtifactError):
            bundle.save(tmp_path / "bad.joblib")

    def test_severity_lookup(self, bundle: ModelBundle):
        assert bundle.severity_for("Normal Traffic") == "Low"
        assert bundle.severity_for("DoS") == "Critical"
        assert bundle.severity_for("Unknown Thing") == "Medium"


class TestLegacyArtifacts:
    def test_consistent_pair_has_no_problems(self):
        model = _tiny_estimator()
        enc = _encoder_for(["DoS", "Normal Traffic", "Port Scanning"])
        assert validate_legacy_artifacts(model, enc) == []

    def test_mismatched_pair_is_flagged(self):
        model = _tiny_estimator()
        enc = _encoder_for(["DoS", "Normal Traffic"])  # 2 classes vs 3
        problems = validate_legacy_artifacts(model, enc)
        assert any("Class-count mismatch" in p for p in problems)

    def test_load_legacy_from_disk(self, bundle: ModelBundle, tmp_path):
        import joblib

        joblib.dump(bundle.estimator, tmp_path / "network_intrusion_detector.pkl")
        joblib.dump(bundle.label_encoder, tmp_path / "label_encoder.pkl")
        loaded = load_legacy_artifacts(tmp_path)
        assert loaded.classes == bundle.classes

    def test_load_legacy_missing_files_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_legacy_artifacts(tmp_path)

    def test_load_legacy_inconsistent_pair_raises(self, tmp_path):
        import joblib

        model = _tiny_estimator()
        enc = _encoder_for(["DoS", "Normal Traffic"])
        joblib.dump(model, tmp_path / "network_intrusion_detector.pkl")
        joblib.dump(enc, tmp_path / "label_encoder.pkl")
        with pytest.raises(ArtifactError, match="inconsistent"):
            load_legacy_artifacts(tmp_path)


class TestFingerprint:
    def test_fingerprint_stable(self, tmp_path, flows):
        path = tmp_path / "data.csv"
        flows.to_csv(path, index=False)
        fp1 = file_fingerprint(path)
        fp2 = file_fingerprint(path)
        assert fp1 == fp2
        assert fp1["rows"] == len(flows)
        assert len(fp1["sha256"]) == 64
