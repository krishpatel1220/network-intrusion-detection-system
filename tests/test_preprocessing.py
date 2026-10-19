import numpy as np
import pandas as pd
import pytest

from config import FEATURE_COLUMNS, TARGET_COLUMN
from src.preprocessing import (
    canonicalize_label_text,
    clean_flows,
    map_raw_labels,
    normalize_class_label,
    split_dataset,
)


class TestLabelNormalization:
    def test_benign_aliases_collapse_to_normal_traffic(self):
        for alias in ("BENIGN", "benign", "Benign Traffic", "normal"):
            assert normalize_class_label(alias) == "Normal Traffic"

    def test_canonicalization_repairs_mojibake_en_dash(self):
        assert (
            canonicalize_label_text("Web Attack \ufffd XSS")
            == "Web Attack \u2013 XSS"
        )
        assert normalize_class_label("Web Attack \ufffd XSS") == "Web Attacks"

    def test_map_raw_labels_full_taxonomy(self):
        series = pd.Series(
            ["BENIGN", "DoS Hulk", "DoS slowloris", "FTP-Patator", "SSH-Patator",
             "PortScan", "Bot", "DDoS", "Heartbleed", "Infiltration", "???"]
        )
        mapped, excluded, unmapped = map_raw_labels(series)
        assert mapped.tolist()[:8] == [
            "Normal Traffic", "DoS", "DoS", "Brute Force", "Brute Force",
            "Port Scanning", "Bots", "DDoS",
        ]
        assert mapped.isna().sum() == 3  # Heartbleed, Infiltration, ???
        assert excluded == {"Heartbleed": 1, "Infiltration": 1}
        assert unmapped == {"???": 1}


class TestCleanFlows:
    def test_removes_duplicates_nans_infs_and_conflicts(self, dirty_flows: pd.DataFrame):
        cleaned, report = clean_flows(dirty_flows, feature_columns=FEATURE_COLUMNS)

        assert report.rows_in == len(dirty_flows) == 194
        assert report.rows_out == len(cleaned)
        assert report.nan_rows_dropped == 6
        assert report.inf_cells_replaced == 3
        assert report.duplicate_rows_dropped == 3
        assert report.conflicting_rows_dropped == 2
        assert report.excluded_label_counts == {"Heartbleed": 1}
        assert report.unmapped_label_counts == {}
        assert report.label_counts == {
            "Normal Traffic": 117,
            "DoS": 40,
            "Port Scanning": 25,
        }
    def test_cleaned_frame_has_no_infs_or_nans(self, dirty_flows: pd.DataFrame):
        cleaned, _ = clean_flows(dirty_flows, feature_columns=FEATURE_COLUMNS)
        values = cleaned[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all()

    def test_schema_is_enforced_and_ordered(self, dirty_flows: pd.DataFrame):
        extra = dirty_flows.copy()
        extra["Some Irrelevant Column"] = 1.0
        cleaned, _ = clean_flows(extra, feature_columns=FEATURE_COLUMNS)
        assert list(cleaned.columns) == list(FEATURE_COLUMNS) + [TARGET_COLUMN]

    def test_missing_target_column_raises(self, flows: pd.DataFrame):
        with pytest.raises(KeyError, match=TARGET_COLUMN):
            clean_flows(flows.drop(columns=[TARGET_COLUMN]))

    def test_missing_feature_columns_raise(self, flows: pd.DataFrame):
        with pytest.raises(KeyError, match="missing required feature"):
            clean_flows(
                flows.drop(columns=["Flow Duration"]), feature_columns=FEATURE_COLUMNS
            )

    def test_label_counts_are_reported(self, flows: pd.DataFrame):
        _, report = clean_flows(flows)
        assert sum(report.label_counts.values()) == report.rows_out
        assert report.label_counts["Normal Traffic"] == 120


class TestSplitting:
    def test_split_sizes_and_stratification(self, flows: pd.DataFrame):
        cleaned, _ = clean_flows(flows, feature_columns=FEATURE_COLUMNS)
        split = split_dataset(cleaned, FEATURE_COLUMNS, random_state=42)

        sizes = split.sizes()
        assert sizes["train"] + sizes["validation"] + sizes["test"] == len(cleaned)
        assert abs(sizes["train"] / len(cleaned) - 0.70) < 0.01

        full_share = cleaned[TARGET_COLUMN].value_counts(normalize=True)
        for part in (split.y_train, split.y_val, split.y_test):
            part_share = part.value_counts(normalize=True)
            assert np.allclose(part_share.sort_index(), full_share.sort_index(), atol=0.02)

    def test_no_feature_overlap_after_deduplication(self, dirty_flows: pd.DataFrame):
        cleaned, _ = clean_flows(dirty_flows, feature_columns=FEATURE_COLUMNS)
        split = split_dataset(cleaned, FEATURE_COLUMNS, random_state=42)
        assert split.train_test_overlap_rows == 0

    def test_split_detects_overlap_when_data_was_not_deduplicated(self, flows: pd.DataFrame):
        duped = pd.concat([flows, flows.head(50)], ignore_index=True)
        split = split_dataset(duped, FEATURE_COLUMNS, random_state=42)
        assert split.train_test_overlap_rows > 0

    def test_invalid_fractions_raise(self, flows: pd.DataFrame):
        with pytest.raises(ValueError):
            split_dataset(flows, FEATURE_COLUMNS, train_fraction=1.5)
        with pytest.raises(ValueError):
            split_dataset(flows, FEATURE_COLUMNS, train_fraction=0.7, val_fraction=0.4)

    def test_deterministic_for_seed(self, flows: pd.DataFrame):
        a = split_dataset(flows, FEATURE_COLUMNS, random_state=42)
        b = split_dataset(flows, FEATURE_COLUMNS, random_state=42)
        assert a.X_train.index.tolist() == b.X_train.index.tolist()
        c = split_dataset(flows, FEATURE_COLUMNS, random_state=1)
        assert a.X_train.index.tolist() != c.X_train.index.tolist()

