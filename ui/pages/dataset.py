from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import DATASET_CHARTS_DIR, TARGET_COLUMN
from src.artifacts import ModelBundle
from ui.components import load_dataset_summary_cached, metric_card, section_title

_CHART_TITLES = {
    "01_attack_type_distribution.png": "Attack type distribution",
    "02_attack_type_composition.png": "Attack type composition",
    "03_flow_duration_boxplot.png": "Flow duration by attack type",
    "04_packet_length_distribution.png": "Packet length distribution",
    "05_packets_vs_bytes_scatter.png": "Packets vs bytes",
    "06_forward_vs_backward_scatter.png": "Forward vs backward traffic",
    "07_tcp_flags_by_attack.png": "TCP flags by attack type",
    "09_correlation_heatmap.png": "Feature correlation heatmap",
    "10_average_packet_size_violin.png": "Average packet size (violin)",
    "12_average_packet_size_scatter.png": "Average packet size vs duration",
}


def render(bundle: ModelBundle) -> None:
    section_title(
        "Dataset Insights",
        "Exploratory analysis of the CICIDS2017 sample used for this project.",
    )

    try:
        summary = load_dataset_summary_cached()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    source_label = (
        "full local processed dataset"
        if summary["full_sample"]
        else "packaged preview (full CSV not present locally)"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Rows", f"{summary['rows']:,}", source_label, accent="cyan")
    with col2:
        metric_card("Columns", str(summary["columns"]), "52 features + label", accent="violet")
    with col3:
        metric_card("Missing values", f"{summary['missing']:,}", "after preparation", accent="teal")
    with col4:
        n_classes = len(summary["class_counts"])
        metric_card("Classes", str(n_classes), "incl. normal traffic", accent="amber")

    tab_preview, tab_dist, tab_cols, tab_gallery = st.tabs(
        ["Preview", "Class distribution", "Column details", "Chart gallery"]
    )

    with tab_preview:
        st.markdown("**First rows of the sample**")
        st.dataframe(summary["preview"], width="stretch", hide_index=True)
        st.caption(
            "Regenerate preview + overview from your own prepared data with "
            "`python prepare_data.py`."
        )

    with tab_dist:
        counts = summary["class_counts"]
        if TARGET_COLUMN in counts.columns and "Records" in counts.columns:
            chart = counts.set_index(TARGET_COLUMN)["Records"]
        else:  # fallback layout safety
            chart = counts.iloc[:, -1]
        st.bar_chart(chart, height=380)
        st.markdown("**Counts**")
        st.dataframe(counts, width="stretch", hide_index=True)

        total = int(counts["Records"].sum()) if "Records" in counts.columns else 0
        normal = (
            int(counts.loc[counts[TARGET_COLUMN] == "Normal Traffic", "Records"].sum())
            if TARGET_COLUMN in counts.columns
            else 0
        )
        if total:
            st.caption(
                f"Benign share: {normal / total:.1%} — the strong class imbalance "
                "is why training uses class-balanced sample weights and reports "
                "macro-F1 alongside accuracy."
            )

    with tab_cols:
        st.dataframe(summary["column_info"], width="stretch", hide_index=True)

    with tab_gallery:
        images = _chart_files()
        if not images:
            st.info(
                f"No chart images found in `{DATASET_CHARTS_DIR.name}/`. They are "
                "optional static EDA outputs shipped with the repository."
            )
        for i in range(0, len(images), 2):
            cols = st.columns(2)
            for col, path in zip(cols, images[i : i + 2]):
                with col:
                    st.image(path, caption=_CHART_TITLES.get(path.name, path.stem))
                    if i + 2 > len(images):
                        break


def _chart_files() -> list[Path]:
    if not DATASET_CHARTS_DIR.exists():
        return []
    return sorted(
        p for p in DATASET_CHARTS_DIR.glob("*.png") if p.name != "08_hidden_file.png"
    )
