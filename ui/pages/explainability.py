from __future__ import annotations
import pandas as pd
import streamlit as st

from config import FULL_DATASET_PATH, SAMPLE_PREVIEW_PATH, TARGET_COLUMN
from src.artifacts import ModelBundle
from src.explainability import global_feature_importance, shap_capable
from ui.components import alert_card, metric_card, section_title

_SAMPLE_ROWS = 500


def render(bundle: ModelBundle) -> None:
    section_title(
        "Explainability",
        "Which flow features drive the deployed model's decisions — and how to read them.",
    )

    if shap_capable(bundle):
        alert_card(
            "info",
            "Values below are <strong>exact SHAP contributions</strong> (TreeSHAP) "
            "computed by the XGBoost booster itself. Each number is the average "
            "absolute contribution of a feature to the model's output — positive "
            "means the feature pushes predictions away from the normal-traffic class.",
        )
    else:
        st.info(
            "The deployed model is not tree-based; showing its built-in feature "
            "importances instead of SHAP values."
        )

    sample = _load_sample(bundle)
    if sample is None:
        st.warning(
            "No sample data available for computing global importance. Provide "
            "`data/processed/cicids2017.csv` (run `python prepare_data.py`) or the "
            "packaged preview file."
        )
        return

    with st.spinner("Computing feature importance ..."):
        try:
            overall = global_feature_importance(bundle, sample)
        except TypeError as exc:
            st.error(str(exc))
            return

    col1, col2, col3 = st.columns(3)
    top = overall.iloc[0]
    with col1:
        metric_card(
            "Sample Size",
            f"{len(sample):,}",
            "flows used for the computation",
            accent="cyan",
        )
    with col2:
        metric_card(
            "Top Feature",
            str(top["Feature"]),
            f"mean |SHAP| = {top['Importance']:.3f}",
            accent="teal",
        )
    with col3:
        metric_card(
            "Method",
            "TreeSHAP" if shap_capable(bundle) else "Built-in",
            "per-flow explanations on other pages",
            accent="violet",
        )

    st.markdown("")
    st.markdown("**Global feature importance (mean absolute contribution)**")

    top_n = st.slider("Number of features to show", 5, 52, 20, key="shap_top_n")
    chart = overall.head(top_n).set_index("Feature")
    st.bar_chart(chart, height=420)

    if shap_capable(bundle):
        _per_class_tab(bundle, sample)

    with st.expander("How to read this chart"):
        st.markdown(
            """
            """
        )


def _per_class_tab(bundle: ModelBundle, sample: pd.DataFrame) -> None:
    with st.expander("Per-class feature importance"):
        with st.spinner("Computing per-class contributions ..."):
            try:
                per_class = global_feature_importance(bundle, sample, per_class=True)
            except Exception as exc:  # pragma: no cover - defensive
                st.error(f"Could not compute per-class importance: {exc}")
                return
        chosen = st.selectbox(
            "Attack class",
            options=sorted(per_class["Class"].unique()),
            index=0,
            key="shap_class",
        )
        subset = per_class[per_class["Class"] == chosen].head(20)
        st.bar_chart(subset.set_index("Feature"), height=400)
        st.caption(
            f"Top features driving predictions toward **{chosen}**, averaged over "
            "the sampled flows."
        )
def _load_sample(bundle: ModelBundle) -> pd.DataFrame | None:
    """Sample flows aligned to the model schema, preferring the full dataset."""
    try:
        if FULL_DATASET_PATH.exists():
            df = pd.read_csv(
                FULL_DATASET_PATH,
                nrows=_SAMPLE_ROWS * 5,
            )
        elif SAMPLE_PREVIEW_PATH.exists():
            df = pd.read_csv(SAMPLE_PREVIEW_PATH)
        else:
            return None
    except Exception:
        return None

    if TARGET_COLUMN in df.columns:
        # Keep some label diversity in the sample: first N rows of each class.
        parts = [
            group.head(max(1, _SAMPLE_ROWS // max(1, df[TARGET_COLUMN].nunique())))
            for _, group in df.groupby(TARGET_COLUMN)
        ]
        df = pd.concat(parts)

    missing = [c for c in bundle.feature_names if c not in df.columns]
    if missing:
        st.error(f"Sample data is missing model features: {', '.join(missing[:5])} ...")
        return None
    sample = df[bundle.feature_names].head(_SAMPLE_ROWS)
    return sample if len(sample) else None