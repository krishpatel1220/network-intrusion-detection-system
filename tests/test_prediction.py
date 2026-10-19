import numpy as np
import pandas as pd
import pytest

from config import FEATURE_COLUMNS, TARGET_COLUMN
from src.prediction import (
    PredictionInputError,
    align_features,
    predict_flows,
    recommendation_for,
    summarize_predictions,
)


class TestAlignFeatures:
    def test_selects_and_orders_schema(self, flows: pd.DataFrame):
        shuffled = flows[list(reversed(FEATURE_COLUMNS)) + [TARGET_COLUMN]]
        aligned = align_features(shuffled, list(FEATURE_COLUMNS))
        assert list(aligned.columns) == list(FEATURE_COLUMNS)

    def test_extra_columns_are_tolerated(self, flows: pd.DataFrame):
        df = flows.copy()
        df["Source IP"] = "10.0.0.1"
        aligned = align_features(df, list(FEATURE_COLUMNS))
        assert "Source IP" not in aligned.columns

    def test_missing_columns_raise_with_names(self, flows: pd.DataFrame):
        df = flows.drop(columns=["Flow Duration", "Destination Port"])
        with pytest.raises(PredictionInputError, match="Flow Duration"):
            align_features(df, list(FEATURE_COLUMNS))

    def test_empty_input_raises(self, flows: pd.DataFrame):
        with pytest.raises(PredictionInputError, match="empty"):
            align_features(flows.iloc[:0], list(FEATURE_COLUMNS))

    def test_non_numeric_values_become_nan(self, flows: pd.DataFrame):
        df = flows.copy()
        # Simulate a dirty CSV read: the column ends up as object dtype.
        df["Flow Duration"] = df["Flow Duration"].astype(object)
        df.loc[0, "Flow Duration"] = "not-a-number"
        aligned = align_features(df, list(FEATURE_COLUMNS))
        assert pd.isna(aligned.loc[0, "Flow Duration"])


class TestPredictFlows:
    def test_predict_flows_adds_columns(self, trained_bundle, flows: pd.DataFrame):
        batch = predict_flows(trained_bundle, flows.head(10))
        result = batch.results
        for col in ("Record Number", "Prediction", "Severity", "Confidence"):
            assert col in result.columns
        assert result["Record Number"].tolist() == list(range(1, 11))
        assert result["Severity"].isin(["Low", "Medium", "High", "Critical"]).all()
        assert result["Confidence"].between(0.0, 1.0).all()

    def test_normal_traffic_gets_low_severity(self, trained_bundle, flows: pd.DataFrame):
        normal = flows[flows[TARGET_COLUMN] == "Normal Traffic"].head(5)
        batch = predict_flows(trained_bundle, normal)
        assert (batch.results["Severity"] == "Low").all()

    def test_probability_frame_classes(self, trained_bundle, flows: pd.DataFrame):
        batch = predict_flows(trained_bundle, flows.head(5))
        assert batch.probabilities is not None
        assert set(batch.probabilities.columns) == set(trained_bundle.classes)
        assert np.allclose(batch.probabilities.sum(axis=1), 1.0, atol=1e-5)

    def test_original_columns_are_preserved(self, trained_bundle, flows: pd.DataFrame):
        batch = predict_flows(trained_bundle, flows.head(3))
        assert TARGET_COLUMN in batch.results.columns


class TestSummaries:
    def _result_frame(self, predictions: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"Prediction": predictions})

    def test_pure_benign_is_healthy(self):
        summary = summarize_predictions(self._result_frame(["Normal Traffic"] * 20))
        assert summary.health == "Healthy"
        assert summary.attack_rate == 0.0
        assert summary.normal_count == 20

    def test_high_attack_rate_is_high_risk(self):
        summary = summarize_predictions(
            self._result_frame(["Normal Traffic", "DoS", "DDoS", "DoS"])
        )
        assert summary.health == "High Risk"
        assert summary.attack_rate == 75.0
        assert summary.recommendation

    def test_medium_band(self):
        summary = summarize_predictions(
            self._result_frame(["Normal Traffic"] * 7 + ["DoS"] * 3)
        )
        assert summary.health == "Medium Risk"

    def test_empty_frame(self):
        summary = summarize_predictions(self._result_frame([]))
        assert summary.total_records == 0
        assert summary.health == "Healthy"


class TestRecommendations:
    def test_all_severities_have_text(self):
        for severity in ("Low", "Medium", "High", "Critical"):
            assert recommendation_for(severity)

    def test_unknown_severity_falls_back(self):
        assert "Review" in recommendation_for("Weird")

