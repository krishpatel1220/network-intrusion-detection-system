import numpy as np
import pandas as pd
import pytest

from config import FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import clean_flows, split_dataset
from src.training import (
    experiment_config,
    model_specs,
    train_all_models,
    train_model,
)


@pytest.fixture(scope="module")
def split():
    from tests.conftest import make_flows

    df = make_flows(n_normal=140, n_dos=50, n_scan=30)
    cleaned, _ = clean_flows(df, feature_columns=FEATURE_COLUMNS)
    return split_dataset(cleaned, FEATURE_COLUMNS, random_state=0)


class TestModelSpecs:
    def test_four_expected_models(self):
        names = [s.name for s in model_specs()]
        assert names == ["logistic_regression", "decision_tree", "random_forest", "xgboost"]

    def test_only_logistic_regression_scales(self):
        scaled = {s.name: s.scale for s in model_specs()}
        assert scaled["logistic_regression"] is True
        assert scaled["xgboost"] is False

    def test_xgboost_uses_sample_weights_and_eval_set(self):
        spec = next(s for s in model_specs() if s.name == "xgboost")
        assert spec.use_sample_weight and spec.supports_eval_set


class TestTrainModel:
    def test_produces_valid_bundle_components(self, split):
        spec = next(s for s in model_specs() if s.name == "logistic_regression")
        trained = train_model(spec, split=split, random_state=0, n_jobs=1)

        assert "validation" in trained.evaluations
        assert "test" in trained.evaluations
        result = trained.evaluations["test"]
        assert result.accuracy > 0.9
        assert result.attack_fpr < 0.1

    def test_classes_are_contiguous_integers(self, split):
        spec = next(s for s in model_specs() if s.name == "decision_tree")
        trained = train_model(spec, split=split, random_state=0, n_jobs=1)
        assert list(trained.estimator.classes_) == [0, 1, 2]
        assert trained.classes == ["DoS", "Normal Traffic", "Port Scanning"]

    def test_reproducible_for_same_seed(self, split):
        spec = next(s for s in model_specs() if s.name == "random_forest")
        a = train_model(spec, split=split, random_state=0, n_jobs=1)
        b = train_model(spec, split=split, random_state=0, n_jobs=1)
        preds_a = a.estimator.predict(split.X_test)
        preds_b = b.estimator.predict(split.X_test)
        assert np.array_equal(preds_a, preds_b)

    def test_requires_training_data(self):
        spec = model_specs()[0]
        with pytest.raises(ValueError, match="DatasetSplit or X_train"):
            train_model(spec)


class TestTrainAllModels:
    def test_all_models_evaluate_on_test(self, split):
        trained = train_all_models(split, random_state=0, n_jobs=1)
        assert [t.name for t in trained] == [
            "logistic_regression", "decision_tree", "random_forest", "xgboost",
        ]
        for t in trained:
            assert "test" in t.evaluations
            assert 0.0 <= t.evaluations["test"].accuracy <= 1.0


class TestExperimentConfig:
    def test_config_captures_reproducibility_metadata(self, split):
        config = experiment_config(split, random_state=7, dataset_info={"path": "x.csv"})
        assert config["random_state"] == 7
        assert config["split_sizes"]["train"] == len(split.X_train)
        assert "scikit-learn" in config["library_versions"]
        assert set(config["class_distribution"]) == {"train", "validation", "test"}
        # The class distribution must cover every label present overall.
        total_labels = set(split.y_train) | set(split.y_val) | set(split.y_test)
        assert set(config["class_distribution"]["train"]) >= {
            str(v) for v in total_labels
        }
