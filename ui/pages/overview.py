from __future__ import annotations
from pathlib import Path
import streamlit as st

from src.artifacts import ModelBundle
from ui.components import (
    class_tags,
    load_model_comparison,
    metric_card,
    production_accuracy,
    section_title,
    tag,
)


def render(bundle: ModelBundle) -> None:
    section_title(
        "Security Operations Overview",
        "Operational intelligence for the deployed intrusion detection workflow.",
    )

    accuracy = production_accuracy(bundle)
    best_label, best_accuracy = _best_from_comparison()
    dataset = _dataset_label(bundle)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(
            "Deployed Model",
            bundle.model_label,
            f"{len(bundle.feature_names)} flow features",
            accent="cyan",
        )
    with col2:
        if accuracy is not None:
            metric_card(
                "Test Accuracy",
                f"{accuracy:.2%}",
                "Held-out test set, recorded at training",
                accent="teal",
            )
        else:
            metric_card("Test Accuracy", "—", "Retrain to record metrics", accent="teal")
    with col3:
        metric_card(
            "Best of 4 Models",
            best_label or "—",
            f"macro-F1 {best_accuracy:.4f}" if best_accuracy is not None else "Train to compare",
            accent="violet",
        )
    with col4:
        metric_card(
            "Traffic Classes",
            str(len(bundle.classes)),
            "Normal traffic + 6 intrusion categories",
            accent="amber",
        )

    st.markdown("")

    tab_overview, tab_pipeline, tab_coverage = st.tabs(
        ["Overview", "Pipeline", "Coverage & Features"]
    )

    with tab_overview:
        from ui.components import alert_card

        alert_card(
            "info",
            "Use the sidebar to move between the <strong>Threat Scanner</strong> "
            "(bulk CSV/Excel/JSON), <strong>Manual Prediction</strong> (single-flow "
            "testing), <strong>Model Performance</strong>, and the new "
            "<strong>Explainability</strong> view.",
        )
        st.markdown(
            f"This platform pairs a trained **{bundle.model_label}** classifier with an "
            "analyst-oriented interface for reviewing network traffic behaviour - upload "
            "a dataset or test a single flow, and get an instant classification with a "
            "severity rating, confidence, and recommended action.\n\n"
            f"Trained on {dataset}."
        )
        st.caption(
            "Every metric shown in this application comes from the recorded "
            "training experiment (see the Model Performance page); nothing is "
            "hard-coded."
        )

    with tab_pipeline:
        steps = [
            ("1", "Load", "The validated model bundle (estimator + encoder + schema) is loaded once per session."),
            ("2", "Validate", "Uploaded traffic is checked against the model's feature schema and aligned to its column order."),
            ("3", "Predict", "Predictions are decoded through the bundle's own label encoder and mapped to a triage severity."),
            ("4", "Report", "Results, confusion analysis, and per-flow SHAP explanations are available for export."),
        ]
        cols = st.columns(4)
        for col, (num, title, desc) in zip(cols, steps):
            with col:
                st.markdown(
                    f"<div class='metric-card accent-cyan' style='text-align:center;'>"
                    f"<div style='font-size:1.6rem;'>{num}</div>"
                    f"<div style='font-weight:700; margin:0.3rem 0;'>{title}</div>"
                    f"<div class='metric-caption'>{desc}</div></div>",
                    unsafe_allow_html=True,
                )

    with tab_coverage:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Supported traffic classes**")
            st.markdown(class_tags(), unsafe_allow_html=True)
        with col2:
            st.markdown("**Application features**")
            st.markdown(
                "- Bulk CSV / Excel / JSON scanning\n"
                "- Ground-truth evaluation when labels are present\n"
                "- Single-flow manual testing with presets\n"
                "- SHAP-based per-flow explanations\n"
                "- Severity mapping and health scoring\n"
                "- Analyst-ready reporting and export"
            )


def _best_from_comparison() -> tuple[str | None, float | None]:
    """Top model by macro-F1 from the recorded experiment, if available."""
    try:
        comparison = load_model_comparison()
    except Exception:
        return None, None
    if comparison.empty or "F1 (macro)" not in comparison.columns:
        return None, None
    row = comparison.loc[comparison["F1 (macro)"].idxmax()]
    return str(row["Model"]), float(row["F1 (macro)"])


def _dataset_label(bundle: ModelBundle) -> str:
    dataset = bundle.training_config.get("dataset", {})
    name = Path(str(dataset.get("path", ""))).stem.replace("cicids2017", "CICIDS2017")
    rows = dataset.get("rows_after_cleaning")
    if rows:
        return f"{name} ({rows:,} cleaned flows)"
    return name or "CICIDS2017"
