from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.artifacts import ModelBundle
from src.explainability import explain_prediction, shap_capable
from src.feature_synthesis import synthesize_manual_flow
from src.prediction import predict_flows, recommendation_for
from ui.components import section_title, severity_badge

#: Realistic starting points for the compact indicator form.
PRESETS: dict[str, dict] = {
    "Normal HTTPS browsing": {
        "destination_port": 443,
        "protocol": "TCP",
        "connection_duration": 12.0,
        "total_packets": 40,
        "average_packet_size": 640,
        "max_packet_size": 1460,
        "flow_bytes_per_second": 2100.0,
        "flow_packets_per_second": 3.4,
    },
    "Web server under DDoS": {
        "destination_port": 80,
        "protocol": "TCP",
        "connection_duration": 0.6,
        "total_packets": 4,
        "average_packet_size": 52,
        "max_packet_size": 60,
        "flow_bytes_per_second": 340.0,
        "flow_packets_per_second": 6.8,
    },
    "Port scan sweep": {
        "destination_port": 22,
        "protocol": "TCP",
        "connection_duration": 0.01,
        "total_packets": 2,
        "average_packet_size": 46,
        "max_packet_size": 60,
        "flow_bytes_per_second": 9200.0,
        "flow_packets_per_second": 200.0,
    },
    "SSH brute-force attempt": {
        "destination_port": 22,
        "protocol": "TCP",
        "connection_duration": 4.5,
        "total_packets": 24,
        "average_packet_size": 130,
        "max_packet_size": 240,
        "flow_bytes_per_second": 690.0,
        "flow_packets_per_second": 5.3,
    },
}

DEFAULT_PRESET = "Normal HTTPS browsing"


def render(bundle: ModelBundle) -> None:
    section_title(
        "Manual Traffic Test",
        "Describe a flow with a few indicators and see how the model classifies it.",
    )

    preset_name = st.selectbox(
        "Preset",
        options=list(PRESETS),
        index=list(PRESETS).index(DEFAULT_PRESET),
        help="Presets fill the form with realistic indicator values.",
    )
    values = dict(PRESETS[preset_name])

    with st.form("manual_prediction_form"):
        row1 = st.columns(4)
        with row1[0]:
            destination_port = st.number_input(
                "Destination Port", 0, 65535, int(values["destination_port"])
            )
        with row1[1]:
            protocol = st.radio(
                "Protocol", ["TCP", "UDP"], horizontal=True,
                index=0 if values["protocol"] == "TCP" else 1,
            )
        with row1[2]:
            connection_duration = st.number_input(
                "Connection Duration (s)", 0.001, 3600.0,
                float(values["connection_duration"]), step=0.1, format="%.3f",
            )
        with row1[3]:
            total_packets = st.number_input(
                "Total Packets", 1, 100000, int(values["total_packets"])
            )

        row2 = st.columns(4)
        with row2[0]:
            average_packet_size = st.number_input(
                "Average Packet Size (bytes)", 1, 1500, int(values["average_packet_size"])
            )
        with row2[1]:
            max_packet_size = st.number_input(
                "Max Packet Size (bytes)", 1, 1500, int(values["max_packet_size"])
            )
        with row2[2]:
            flow_bytes_per_second = st.number_input(
                "Flow Bytes / Second", 0.0, 1_000_000.0,
                float(values["flow_bytes_per_second"]), step=10.0,
            )
        with row2[3]:
            flow_packets_per_second = st.number_input(
                "Flow Packets / Second", 0.0, 100_000.0,
                float(values["flow_packets_per_second"]), step=0.1,
            )

        submitted = st.form_submit_button("Classify this flow", type="primary")

    if not submitted:
        st.caption(
            f"The form collects {8} indicators; the full "
            f"{len(bundle.feature_names)}-feature vector is derived from them "
            "(directional packet splits, inter-arrival statistics, window sizes) "
            "so the input lands in a realistic region of feature space. For exact "
            "recordings, use the Threat Scanner instead."
        )
        return

    indicators = {
        "destination_port": destination_port,
        "protocol": protocol,
        "connection_duration": connection_duration,
        "total_packets": total_packets,
        "average_packet_size": average_packet_size,
        "max_packet_size": max_packet_size,
        "flow_bytes_per_second": flow_bytes_per_second,
        "flow_packets_per_second": flow_packets_per_second,
    }

    try:
        features = synthesize_manual_flow(indicators, bundle.feature_names)
    except (KeyError, ValueError) as exc:
        st.error(f"Could not build a flow from the form: {exc}")
        return

    with st.spinner("Classifying ..."):
        batch = predict_flows(bundle, features)
    result = batch.results.iloc[0]

    _render_result(bundle, result, features, batch.probabilities)


def _render_result(
    bundle: ModelBundle,
    result: pd.Series,
    features: pd.DataFrame,
    probabilities: pd.DataFrame | None = None,
) -> None:
    prediction = str(result["Prediction"])
    severity = str(result["Severity"])

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            f"<div class='result-title'>{prediction}</div>"
            + severity_badge(severity),
            unsafe_allow_html=True,
        )
        if "Confidence" in result.index and pd.notna(result["Confidence"]):
            st.markdown(
                f"<div class='confidence-value'>{float(result['Confidence']):.1%}</div>",
                unsafe_allow_html=True,
            )
            st.caption("Model confidence (max class probability)")
        st.markdown(
            f"<div class='recommendation-box'>{recommendation_for(severity)}</div>",
            unsafe_allow_html=True,
        )

        if probabilities is not None and len(probabilities):
            _render_probability_chips(probabilities.iloc[0])

    with col2:
        _render_explanation(bundle, features, prediction)

    st.markdown("")
    _render_derived_features(features, bundle)


def _render_probability_chips(probs_row: pd.Series) -> None:
    """Per-class probability chips, sorted by likelihood."""
    chips = " ".join(
        f"<span class='tag tag-cyan'>{name}: {float(value):.1%}</span>"
        for name, value in sorted(probs_row.items(), key=lambda kv: -float(kv[1]))
    )
    st.markdown(chips, unsafe_allow_html=True)


def _render_explanation(bundle: ModelBundle, features: pd.DataFrame, prediction: str) -> None:
    if not shap_capable(bundle):
        st.info("Per-flow SHAP explanation requires the XGBoost production model.")
        return

    explanation = explain_prediction(bundle, features, top_n=8)
    if explanation is None:
        return

    st.markdown("**Top reasons for this decision**")
    contrib = explanation.contributions.sort_values("Contribution")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    colors = ["#0d9488" if v >= 0 else "#e11d48" for v in contrib["Contribution"]]
    ax.barh(contrib["Feature"], contrib["Contribution"], color=colors)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel("SHAP contribution")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        f"Predicted `{explanation.predicted_class}` — base {explanation.base_value:.3f} "
        f"+ contributions = {explanation.total:.3f} (margin score)."
    )


def _render_derived_features(features: pd.DataFrame, bundle: ModelBundle) -> None:
    with st.expander("Derived feature vector (all 52 model inputs)"):
        st.dataframe(
            features.T.rename(columns={0: "value"}), width="stretch"
        )
        st.download_button(
            "Download this flow as CSV",
            features.to_csv(index=False).encode("utf-8"),
            file_name="manual_flow.csv",
            mime="text/csv",
            key="dl_manual_flow",
        )
