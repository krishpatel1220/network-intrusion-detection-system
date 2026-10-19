from __future__ import annotations

import pandas as pd


def synthesize_manual_flow(values: dict, feature_names: list[str]) -> pd.DataFrame:
    required = (
        "destination_port",
        "protocol",
        "connection_duration",
        "total_packets",
        "average_packet_size",
        "max_packet_size",
        "flow_bytes_per_second",
        "flow_packets_per_second",
    )
    missing = [k for k in required if k not in values]
    if missing:
        raise KeyError(f"Missing manual-input indicator(s): {', '.join(missing)}")

    try:
        destination_port = float(values["destination_port"])
        connection_duration = max(0.001, float(values["connection_duration"]))
        total_packets = max(1.0, float(values["total_packets"]))
        average_packet_size = max(1.0, float(values["average_packet_size"]))
        flow_bytes_per_second = float(values["flow_bytes_per_second"])
        flow_packets_per_second = float(values["flow_packets_per_second"])
        max_packet_size = max(average_packet_size, float(values["max_packet_size"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid manual-input indicator: {exc}") from exc

    protocol = str(values["protocol"]).upper()
    is_tcp = protocol == "TCP"

    # Packet split between directions (heuristic 60/40 forward/backward).
    total_fwd_packets = max(1.0, round(total_packets * 0.6))
    total_bwd_packets = max(1.0, round(total_packets * 0.4))
    total_length_fwd_packets = total_fwd_packets * average_packet_size
    total_length_bwd_packets = total_bwd_packets * average_packet_size * 0.7

    # Forward packet-length statistics.
    fwd_packet_length_min = average_packet_size * 0.5
    fwd_packet_length_mean = average_packet_size
    fwd_packet_length_std = max(1.0, (max_packet_size - fwd_packet_length_min) / 6.0)

    # Backward packet-length statistics.
    bwd_packet_length_mean = average_packet_size * 0.7
    bwd_packet_length_max = max_packet_size * 0.75
    bwd_packet_length_min = average_packet_size * 0.4
    bwd_packet_length_std = max(
        1.0, (bwd_packet_length_max - bwd_packet_length_min) / 6.0
    )

    # Inter-arrival time statistics, anchored on duration / packets.
    flow_iat_mean = max(1.0, connection_duration / total_packets)
    flow_iat_std = max(1.0, flow_iat_mean * 0.3)

    fwd_iat_mean = max(1.0, flow_iat_mean * 0.9)
    bwd_iat_mean = max(1.0, flow_iat_mean * 1.05)

    # TCP flag counts.
    ack_flag_count = total_packets if is_tcp else 0.0
    psh_flag_count = total_fwd_packets * 0.5 if is_tcp else 0.0
    fin_flag_count = 1.0 if is_tcp else 0.0
    syn_flag_count = 2.0 if is_tcp else 0.0

    packet_length_std = max(1.0, (max_packet_size - average_packet_size) / 6.0)

    data = {
        "Destination Port": destination_port,
        "Flow Duration": connection_duration,
        "Total Fwd Packets": total_fwd_packets,
        "Total Length of Fwd Packets": total_length_fwd_packets,
        "Fwd Packet Length Max": max_packet_size,
        "Fwd Packet Length Min": fwd_packet_length_min,
        "Fwd Packet Length Mean": fwd_packet_length_mean,
        "Fwd Packet Length Std": fwd_packet_length_std,
        "Bwd Packet Length Max": bwd_packet_length_max,
        "Bwd Packet Length Min": bwd_packet_length_min,
        "Bwd Packet Length Mean": bwd_packet_length_mean,
        "Bwd Packet Length Std": bwd_packet_length_std,
        "Flow Bytes/s": flow_bytes_per_second,
        "Flow Packets/s": flow_packets_per_second,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_mean * 1.5,
        "Flow IAT Min": max(1.0, flow_iat_mean * 0.5),
        "Fwd IAT Total": flow_iat_mean * total_fwd_packets * 0.8,
        "Fwd IAT Mean": fwd_iat_mean,
        "Fwd IAT Std": max(1.0, fwd_iat_mean * 0.25),
        "Fwd IAT Max": fwd_iat_mean * 1.4,
        "Fwd IAT Min": max(1.0, fwd_iat_mean * 0.6),
        "Bwd IAT Total": flow_iat_mean * total_bwd_packets * 0.8,
        "Bwd IAT Mean": bwd_iat_mean,
        "Bwd IAT Std": max(1.0, bwd_iat_mean * 0.25),
        "Bwd IAT Max": bwd_iat_mean * 1.4,
        "Bwd IAT Min": max(1.0, bwd_iat_mean * 0.6),
        "Fwd Header Length": total_fwd_packets * 20.0,
        "Bwd Header Length": total_bwd_packets * 20.0,
        "Fwd Packets/s": flow_packets_per_second * 0.6,
        "Bwd Packets/s": flow_packets_per_second * 0.4,
        "Min Packet Length": average_packet_size * 0.4,
        "Max Packet Length": max_packet_size,
        "Packet Length Mean": average_packet_size,
        "Packet Length Std": packet_length_std,
        "Packet Length Variance": packet_length_std ** 2,
        "FIN Flag Count": fin_flag_count,
        "SYN Flag Count": syn_flag_count,
        "RST Flag Count": 0.0,
        "PSH Flag Count": psh_flag_count,
        "ACK Flag Count": ack_flag_count,
        "URG Flag Count": 0.0,
        "CWE Flag Count": 0.0,
        "ECE Flag Count": 0.0,
        "Average Packet Size": average_packet_size,
        "Subflow Fwd Bytes": total_length_fwd_packets,
        "Init_Win_bytes_forward": 29200.0 if is_tcp else 8192.0,
        "Init_Win_bytes_backward": 29200.0 if is_tcp else 8192.0,
        "act_data_pkt_fwd": max(1, round(total_fwd_packets * 0.5)),
        "min_seg_size_forward": 32.0,
        "Active Mean": connection_duration * 0.6,
        "Active Max": connection_duration * 0.8,
        "Active Min": connection_duration * 0.3,
        "Idle Mean": connection_duration * 0.1,
        "Idle Max": connection_duration * 0.2,
        "Idle Min": connection_duration * 0.02,
    }
    base = {name: 0.0 for name in feature_names}
    base.update({k: v for k, v in data.items() if k in base})
    return pd.DataFrame([base], columns=feature_names)
