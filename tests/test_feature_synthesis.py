import numpy as np
import pandas as pd
import pytest

from config import FEATURE_COLUMNS
from src.feature_synthesis import synthesize_manual_flow

INDICATORS = {
    "destination_port": 443,
    "protocol": "TCP",
    "connection_duration": 3.0,
    "total_packets": 20,
    "average_packet_size": 400,
    "max_packet_size": 1200,
    "flow_bytes_per_second": 3000.0,
    "flow_packets_per_second": 8.0,
}


class TestSynthesizeManualFlow:
    def test_output_matches_schema_exactly(self):
        frame = synthesize_manual_flow(INDICATORS, list(FEATURE_COLUMNS))
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        assert len(frame) == 1

    def test_values_are_finite_numbers(self):
        frame = synthesize_manual_flow(INDICATORS, list(FEATURE_COLUMNS))
        values = frame.to_numpy(dtype=np.float64)
        assert np.isfinite(values).all()

    def test_udp_zeroes_tcp_flags(self):
        udp = dict(INDICATORS, protocol="UDP")
        frame = synthesize_manual_flow(udp, list(FEATURE_COLUMNS))
        assert frame["ACK Flag Count"].iloc[0] == 0.0
        assert frame["PSH Flag Count"].iloc[0] == 0.0

    def test_tcp_sets_flags(self):
        frame = synthesize_manual_flow(INDICATORS, list(FEATURE_COLUMNS))
        assert frame["ACK Flag Count"].iloc[0] > 0
        assert frame["PSH Flag Count"].iloc[0] > 0

    def test_packet_totals_positive(self):
        frame = synthesize_manual_flow(INDICATORS, list(FEATURE_COLUMNS))
        assert frame["Total Fwd Packets"].iloc[0] >= 1
        assert frame["Flow Duration"].iloc[0] > 0

    def test_missing_indicator_raises(self):
        incomplete = {k: v for k, v in INDICATORS.items() if k != "protocol"}
        with pytest.raises(KeyError, match="protocol"):
            synthesize_manual_flow(incomplete, list(FEATURE_COLUMNS))

    def test_invalid_number_raises(self):
        bad = dict(INDICATORS, connection_duration="not-a-number")
        with pytest.raises(ValueError, match="Invalid manual-input"):
            synthesize_manual_flow(bad, list(FEATURE_COLUMNS))

    def test_unknown_schema_features_default_to_zero(self):
        extended = list(FEATURE_COLUMNS) + ["Some Future Feature"]
        frame = synthesize_manual_flow(INDICATORS, extended)
        assert frame["Some Future Feature"].iloc[0] == 0.0

    def test_extreme_values_do_not_crash(self):
        extreme = dict(INDICATORS, total_packets=1, average_packet_size=1, max_packet_size=1)
        frame = synthesize_manual_flow(extreme, list(FEATURE_COLUMNS))
        assert np.isfinite(frame.to_numpy(dtype=np.float64)).all()
