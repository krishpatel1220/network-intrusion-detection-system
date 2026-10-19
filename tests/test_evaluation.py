import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    comparison_table,
    confusion_pairs,
    evaluate_classifier,
)

CLASSES = ["Normal Traffic", "DoS", "Port Scanning"]


class TestEvaluateClassifier:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 2, 2])
        result = evaluate_classifier(y, y.copy(), CLASSES, model_name="m")
        assert result.accuracy == 1.0
        assert result.f1_macro == 1.0
        assert result.attack_fpr == 0.0

    def test_attack_fpr_counts_benign_false_alarms(self):
        # 4 benign: 2 flagged as attack -> FPR 0.5
        y_true = np.array([0, 0, 0, 0, 1, 2])
        y_pred = np.array([0, 0, 1, 2, 1, 2])
        result = evaluate_classifier(y_true, y_pred, CLASSES)
        assert result.attack_fpr == pytest.approx(0.5)

    def test_all_benign_predicted_as_attack(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 2])
        result = evaluate_classifier(y_true, y_pred, CLASSES)
        assert result.attack_fpr == 1.0
        assert result.accuracy == 0.0

    def test_per_class_table_contents(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        result = evaluate_classifier(y_true, y_pred, CLASSES)
        dos = result.per_class.set_index("Class").loc["DoS"]
        # DoS: predicted 3x, correct 2x, one benign flow flagged as DoS.
        assert dos["Support"] == 2
        assert dos["Precision"] == pytest.approx(2 / 3)
        assert dos["Recall"] == pytest.approx(1.0)

    def test_confusion_matrix_labels(self):
        y = np.array([0, 1, 2])
        result = evaluate_classifier(y, y.copy(), CLASSES)
        assert list(result.confusion.index) == [f"true: {c}" for c in CLASSES]
        assert list(result.confusion.columns) == [f"pred: {c}" for c in CLASSES]
        assert int(np.trace(result.confusion.to_numpy())) == 3

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            evaluate_classifier(np.array([0, 1]), np.array([0]), CLASSES)

    def test_empty_predictions_raise(self):
        with pytest.raises(ValueError, match="empty"):
            evaluate_classifier(np.array([]), np.array([]), CLASSES)

    def test_empty_class_list_raises(self):
        with pytest.raises(ValueError, match="class_labels"):
            evaluate_classifier(np.array([0]), np.array([0]), [])


class TestComparisonTableAndPairs:
    def test_comparison_table_sorted_by_macro_f1(self):
        perfect = evaluate_classifier(np.array([0, 1]), np.array([0, 1]), CLASSES, model_name="good")
        bad = evaluate_classifier(np.array([0, 1]), np.array([1, 0]), CLASSES, model_name="bad")
        table = comparison_table([bad, perfect])
        assert table.iloc[0]["Model"] == "good"
        assert set(table.columns) >= {"Model", "Attack FPR", "F1 (macro)"}

    def test_confusion_pairs_finds_off_diagonal(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 2])
        result = evaluate_classifier(y_true, y_pred, CLASSES)
        pairs = confusion_pairs(result)
        top = pairs.iloc[0]
        assert top["True class"] == "Normal Traffic"
        assert top["Predicted as"] == "DoS"
        assert top["Count"] == 1

    def test_confusion_pairs_empty_when_perfect(self):
        y = np.array([0, 1, 2])
        result = evaluate_classifier(y, y.copy(), CLASSES)
        assert confusion_pairs(result).empty
