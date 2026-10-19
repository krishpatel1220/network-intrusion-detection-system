import json

import pandas as pd
import pytest

from config import TARGET_COLUMN
from src.data import FlowFileError, detect_file_type, load_dataset_summary, load_flow_file


class TestDetectFileType:
    def test_supported_extensions(self):
        assert detect_file_type("traffic.csv") == "csv"
        assert detect_file_type("traffic.XLSX") == "xlsx"
        assert detect_file_type("traffic.json") == "json"

    def test_unsupported_extension_raises(self):
        with pytest.raises(FlowFileError, match="Unsupported file type"):
            detect_file_type("traffic.parquet")


class TestLoadFlowFile:
    def test_loads_csv_from_path(self, tmp_path, flows):
        path = tmp_path / "flows.csv"
        flows.to_csv(path, index=False)
        loaded = load_flow_file(path)
        assert len(loaded) == len(flows)
        assert TARGET_COLUMN in loaded.columns

    def test_loads_excel(self, tmp_path, flows):
        pytest.importorskip("openpyxl")
        path = tmp_path / "flows.xlsx"
        flows.to_excel(path, index=False)
        loaded = load_flow_file(path)
        assert len(loaded) == len(flows)

    def test_loads_json_from_file_object(self, flows):
        import io

        buffer = io.BytesIO(flows.to_json(orient="records").encode())
        loaded = load_flow_file(buffer, filename="flows.json")
        assert len(loaded) == len(flows)

    def test_empty_csv_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("a,b\n", encoding="utf-8")
        with pytest.raises(FlowFileError, match="no rows"):
            load_flow_file(path)

    def test_malformed_csv_raises_clear_error(self, tmp_path):
        path = tmp_path / "broken.csv"
        path.write_text('Destination Port,Flow Duration\n"unterminated,1\n', encoding="utf-8")
        with pytest.raises(FlowFileError, match="Could not parse"):
            load_flow_file(path)

    def test_file_object_without_name_raises(self):
        import io

        with pytest.raises(FlowFileError, match="file name is required"):
            load_flow_file(io.BytesIO(b"a,b\n1,2\n"))


class TestDatasetSummary:
    def test_falls_back_to_packaged_assets(self, tmp_path):
        preview = tmp_path / "preview.csv"
        overview_path = tmp_path / "overview.json"
        preview.write_text("a,b\n1,2\n", encoding="utf-8")
        overview_path.write_text(
            json.dumps(
                {
                    "rows": 100,
                    "columns": 2,
                    "missing": 0,
                    "column_info": [{"Column": "a", "Data Type": "int64", "Missing": 0}],
                    "class_counts": [{"Attack Type": "Normal Traffic", "Records": 100}],
                }
            ),
            encoding="utf-8",
        )
        summary = load_dataset_summary(
            dataset_path=tmp_path / "missing.csv",
            preview_path=preview,
            overview_path=overview_path,
        )
        assert summary["rows"] == 100
        assert summary["full_sample"] is False
        assert len(summary["preview"]) == 1

    def test_streams_full_csv(self, tmp_path, flows):
        dataset = tmp_path / "full.csv"
        flows.to_csv(dataset, index=False)
        summary = load_dataset_summary(
            dataset_path=dataset,
            preview_path=tmp_path / "none.csv",
            overview_path=tmp_path / "none.json",
        )
        assert summary["rows"] == len(flows)
        assert summary["full_sample"] is True
        assert summary["missing"] == 0
        counts = summary["class_counts"].set_index("Attack Type")["Records"]
        assert int(counts["Normal Traffic"]) == 120

    def test_missing_everything_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset_summary(
                dataset_path=tmp_path / "a.csv",
                preview_path=tmp_path / "b.csv",
                overview_path=tmp_path / "c.json",
            )
