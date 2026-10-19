import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, TARGET_COLUMN
from src.explainability import (
    explain_prediction,
    global_feature_importance,
    shap_capable,
)
from src.prediction import predict_flows


class TestShapCapability:
    def test_xgboost_is_shap_capable(self, trained_bundle):
        assert shap_capable(trained_bundle) is True

    def test_sklearn_pipeline_without_booster_is_not(self, trained_bundle, flows):
        from sklearn.linear_model import LogisticRegression

        from src.artifacts import ModelBundle

        # Build a *separate* bundle so the shared fixture is untouched.
        # Mix classes: the first rows of make_flows are all benign.
        sample = pd.concat(
            [
                flows[flows[TARGET_COLUMN] == "Normal Traffic"].head(15),
                flows[flows[TARGET_COLUMN] == "DoS"].head(15),
            ]
        )
        lr_bundle = ModelBundle(
            model_name="lr",
            model_label="Logistic Regression",
            estimator=LogisticRegression(max_iter=500).fit(
                sample[list(FEATURE_COLUMNS)],
                (sample[TARGET_COLUMN] == "Normal Traffic").astype(int),
            ),
            label_encoder=trained_bundle.label_encoder,
            feature_names=list(FEATURE_COLUMNS),
        )
        assert shap_capable(lr_bundle) is False


class TestLocalExplanation:
    def test_explain_prediction_matches_predicted_class(self, trained_bundle, flows):
        row = flows.head(1)
        batch = predict_flows(trained_bundle, row)
        predicted = batch.results["Prediction"].iloc[0]

        explanation = explain_prediction(trained_bundle, row[list(FEATURE_COLUMNS)])
        assert explanation is not None
        assert explanation.predicted_class == predicted

    def test_contributions_have_expected_shape(self, trained_bundle, flows):
        row = flows.head(1)[list(FEATURE_COLUMNS)]
        explanation = explain_prediction(trained_bundle, row, top_n=10)
        assert len(explanation.contributions) == 10
        assert list(explanation.contributions.columns) == ["Feature", "Contribution"]
        # Sorted by absolute contribution, descending.
        magnitudes = explanation.contributions["Contribution"].abs().tolist()
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_rejects_multiple_rows(self, trained_bundle, flows):
        import pytest

        with pytest.raises(ValueError, match="exactly one row"):
            explain_prediction(trained_bundle, flows.head(3)[list(FEATURE_COLUMNS)])


class TestGlobalImportance:
    def test_returns_all_features_sorted(self, trained_bundle, flows):
        sample = flows.head(20)[list(FEATURE_COLUMNS)]
        importance = global_feature_importance(trained_bundle, sample)
        assert len(importance) == len(FEATURE_COLUMNS)
        values = importance["Importance"].tolist()
        assert values == sorted(values, reverse=True)
        assert (importance["Importance"] >= 0).all()

    def test_per_class_frame(self, trained_bundle, flows):
        sample = flows.head(20)[list(FEATURE_COLUMNS)]
        importance = global_feature_importance(trained_bundle, sample, per_class=True)
        assert set(importance.columns) == {"Class", "Feature", "Importance"}
        assert set(importance["Class"]) == set(trained_bundle.classes)

    def test_empty_sample_returns_empty(self, trained_bundle, flows):
        empty = flows.head(0)[list(FEATURE_COLUMNS)]
        assert global_feature_importance(trained_bundle, empty).empty


class TestSeparabilityReflectedInShap:
    def test_dos_and_normal_explanations_differ(self, trained_bundle, flows):
        normal = flows[flows[TARGET_COLUMN] == "Normal Traffic"].head(1)
        dos = flows[flows[TARGET_COLUMN] == "DoS"].head(1)
        exp_normal = explain_prediction(trained_bundle, normal[list(FEATURE_COLUMNS)])
        exp_dos = explain_prediction(trained_bundle, dos[list(FEATURE_COLUMNS)])
        assert exp_normal is not None and exp_dos is not None
        top_normal = set(exp_normal.contributions["Feature"].head(3))
        top_dos = set(exp_dos.contributions["Feature"].head(3))
        # The synthetic classes differ mainly in port/duration/size, so the
        # dominant features should not be identical for both classes.
        assert top_normal != top_dos
