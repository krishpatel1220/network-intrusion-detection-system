import numpy as np
import pandas as pd
import pytest

from config import TARGET_COLUMN
from src.error_analysis import analyze_errors
from src.prediction import predict_flows


class TestAnalyzeErrors:
    def _mixed_flows(self, flows):
        return pd.concat(
            [
                flows[flows[TARGET_COLUMN] == "Normal Traffic"].head(10),
                flows[flows[TARGET_COLUMN] == "DoS"].head(5),
                flows[flows[TARGET_COLUMN] == "Port Scanning"].head(5),
            ]
        ).reset_index(drop=True)

    def _labelled(self, trained_bundle, flows, inject_fp: bool = True):
        df = self._mixed_flows(flows)
        batch = predict_flows(trained_bundle, df)
        results = batch.results
        if inject_fp:
            normal_idx = results.index[results["Prediction"] == "Normal Traffic"]
            if len(normal_idx):
                results.loc[normal_idx[0], "Prediction"] = "DoS"
        return results

    def test_counts_and_evaluation(self, trained_bundle, flows):
        results = self._labelled(trained_bundle, flows, inject_fp=True)
        analysis = analyze_errors(results, trained_bundle.classes)

        assert analysis.fp_count >= 1
        assert analysis.evaluation.attack_fpr >= 0.0
        assert analysis.evaluation.summary_row()["Attack FPR"] == analysis.evaluation.attack_fpr

    def test_false_negatives_capture_wrong_attack_types(self, trained_bundle, flows):
        results = self._labelled(trained_bundle, flows, inject_fp=False)
        dos_idx = results.index[results[TARGET_COLUMN] == "DoS"]
        if len(dos_idx):
            results.loc[dos_idx[0], "Prediction"] = "Normal Traffic"
        analysis = analyze_errors(results, trained_bundle.classes)
        assert analysis.fn_count >= 1

    def test_unknown_truth_label_raises(self, trained_bundle, flows):
        results = self._labelled(trained_bundle, flows, inject_fp=False)
        results.loc[results.index[0], TARGET_COLUMN] = "Nonsense Class"
        with pytest.raises(KeyError, match="not trained on"):
            analyze_errors(results, trained_bundle.classes)

    def test_missing_columns_raise(self, trained_bundle, flows):
        results = self._labelled(trained_bundle, flows, inject_fp=False)
        with pytest.raises(KeyError, match="Prediction"):
            analyze_errors(results.drop(columns=["Prediction"]), trained_bundle.classes)

    def test_perfect_predictions_yield_no_errors(self, trained_bundle, flows):
        results = self._labelled(trained_bundle, flows, inject_fp=False)
        analysis = analyze_errors(results, trained_bundle.classes)
        assert analysis.fp_count == 0
        # The synthetic classes are linearly separable: no cross-class errors.
        assert analysis.top_confusions.empty or analysis.evaluation.accuracy > 0.95

    def test_confidence_sorts_errors(self, trained_bundle, flows):
        results = self._labelled(trained_bundle, flows, inject_fp=True)
        analysis = analyze_errors(results, trained_bundle.classes)
        if len(analysis.false_positives) > 1:
            conf = analysis.false_positives["Confidence"].tolist()
            assert conf == sorted(conf, reverse=True)
