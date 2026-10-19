from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config import REPORTS_DIR
from src.artifacts import ModelBundle
from ui.components import load_model_comparison, metric_card, section_title

_DISPLAY_COLUMNS = [
    "Model",
    "Accuracy",
    "Precision (macro)",
    "Recall (macro)",
    "F1 (macro)",
    "F1 (weighted)",
    "Attack FPR",
]

_LEGACY_COLUMNS = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]


def render(bundle: ModelBundle) -> None:
    section_title(
        "Model Performance",
        "Recorded results from the training experiment — never hard-coded values.",
    )

    try:
        comparison = load_model_comparison()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
        return

    columns = [c for c in _DISPLAY_COLUMNS if c in comparison.columns]
    legacy = not columns
    if legacy:
        columns = [c for c in _LEGACY_COLUMNS if c in comparison.columns]
        st.warning(
            "This comparison table was written by the legacy training script and "
            "lacks macro-F1 and false-positive-rate columns. Run `python train.py` "
            "to regenerate it with the full metric suite."
        )

    table = comparison[columns].copy()
    for col in table.columns:
        if col != "Model" and pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(
                lambda v: f"{v:.4f}" if col != "Accuracy" else f"{v:.4%}"
            )

    best_idx = comparison[columns[1]].idxmax() if len(columns) > 1 else None

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(
            "Models Compared",
            str(len(comparison)),
            "Logistic Regression, Decision Tree, Random Forest, XGBoost",
            accent="cyan",
        )
    with col2:
        if best_idx is not None:
            metric_card(
                "Top Model",
                str(comparison.loc[best_idx, "Model"]),
                f"{columns[1]} = {comparison.loc[best_idx, columns[1]]:.4f}",
                accent="teal",
            )
        else:
            metric_card("Top Model", "—", "", accent="teal")
    with col3:
        deployed = bundle.model_label
        metric_card("Deployed", deployed, "served by this application", accent="violet")

    st.markdown("")
    st.markdown("**Comparison (test split)**")
    st.dataframe(table, width="stretch", hide_index=True)

    if "Attack FPR" in comparison.columns:
        _fpr_chart(comparison)
    elif "F1 Score" in comparison.columns:
        st.bar_chart(comparison.set_index("Model")[["F1 Score", "Accuracy"]], height=340)

    _latest_experiment_details()

    with st.expander("Metric definitions — read before quoting these numbers"):
        st.markdown(
            """
            """
        )
def _fpr_chart(comparison: pd.DataFrame) -> None:
    plot = comparison.set_index("Model")[["F1 (macro)", "Attack FPR"]].astype(float)
    st.markdown("**Macro-F1 vs benign false-positive rate**")
    st.bar_chart(plot, height=340)
    st.caption(
        "Attack FPR is shown on the same axis: the best model maximises macro-F1 "
        "while keeping the false-alarm rate near zero."
    )
def _latest_experiment_details() -> None:
    run_dir = _latest_run()
    if run_dir is None:
        return

    with st.expander(f"Latest experiment: `{run_dir.name}` (reports/)"):
        config_path = run_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            st.markdown(
                f"- Seed: `{config.get('random_state')}`  \n"
                f"- Split: {config.get('split_sizes')}  \n"
                f"- Train/test duplicate overlap: "
                f"{config.get('train_test_overlap_rows')} rows  \n"
                f"- Generated: {config.get('generated_at')}"
            )
            st.json(
                {
                    k: config[k]
                    for k in ("dataset", "library_versions")
                    if k in config
                },
                expanded=False,
            )
        report_path = run_dir / "REPORT.md"
        if report_path.exists():
            st.download_button(
                "Download full experiment report (Markdown)",
                data=report_path.read_text(encoding="utf-8"),
                file_name=f"{run_dir.name}_REPORT.md",
                mime="text/markdown",
            )
def _latest_run() -> Path | None:
    if not REPORTS_DIR.exists():
        return None
    runs = sorted(
        (p for p in REPORTS_DIR.iterdir() if p.is_dir() and p.name.startswith("experiment_")),
        key=lambda p: p.name,
    )
    return runs[-1] if runs else None