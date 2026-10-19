from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    CLASS_LABELS,
    MODELS_DIR,
    PRODUCTION_MODEL_NAME,
    MODEL_BUNDLE_SUFFIX,
)
from src.artifacts import ModelBundle, bundle_path_for

# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading trained model ...")
def load_production_bundle(
    model_name: str = PRODUCTION_MODEL_NAME,
) -> ModelBundle:
    """Load and validate the deployed model bundle (cached per session).

    Raises:
        FileNotFoundError: If no bundle exists for *model_name*.
    """
    path = bundle_path_for(MODELS_DIR, model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle '{path.name}' not found in models/. "
            "Run `python train.py` to train and save models."
        )
    return ModelBundle.load(path)


@st.cache_data(show_spinner=False)
def load_model_comparison(_models_dir: str = str(MODELS_DIR)) -> pd.DataFrame:
    """Load the model-comparison table written by ``train.py``."""
    path = Path(_models_dir) / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python train.py` to generate it."
        )
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading dataset summary ...")
def load_dataset_summary_cached() -> dict:
    """Load the bundled CICIDS2017 sample summary (full CSV or fallback)."""
    from src.data import load_dataset_summary

    return load_dataset_summary()


def production_metrics(bundle: ModelBundle) -> dict:
    """Test-set metrics for the deployed model, if recorded."""
    return bundle.metrics.get("test", bundle.metrics.get("validation", {}))


def production_accuracy(bundle: ModelBundle) -> float | None:
    metrics = production_metrics(bundle)
    return metrics.get("accuracy")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def section_title(title: str, subtitle: str = "") -> None:
    """Gradient-underlined section header."""
    if subtitle:
        st.markdown(
            f"<div class='section-header'>{title}</div>"
            f"<div class='subtle'>{subtitle}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


def metric_card(title: str, value: str, caption: str, accent: str = "cyan") -> None:
    """Accent-topped metric card."""
    st.markdown(
        f"<div class='metric-card accent-{accent}'><div class='metric-title'>{title}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-caption'>{caption}</div></div>",
        unsafe_allow_html=True,
    )


def tag(text: str, color: str = "cyan") -> str:
    """Colored pill chip (as an HTML string)."""
    return f"<span class='tag tag-{color}'>{text}</span>"


def severity_badge(severity: str) -> str:
    """Colored severity badge (as an HTML string)."""
    color_class = {
        "Low": "badge-low",
        "Medium": "badge-medium",
        "High": "badge-high",
        "Critical": "badge-critical",
    }.get(severity, "badge-medium")
    return f"<span class='badge {color_class}'>{severity}</span>"


def alert_card(kind: str, html_body: str) -> None:
    """Render one of the themed alert cards (info/success/warning/danger)."""
    css_class = {
        "info": "info-card",
        "success": "success-card",
        "warning": "warning-card",
        "danger": "danger-card",
    }.get(kind, "info-card")
    st.markdown(
        f"<div class='{css_class}'>{html_body}</div>", unsafe_allow_html=True
    )


CLASS_TAG_COLORS = {
    "Normal Traffic": "teal",
    "DoS": "rose",
    "DDoS": "rose",
    "Port Scanning": "amber",
    "Bots": "amber",
    "Brute Force": "violet",
    "Web Attacks": "violet",
}


def class_tags() -> str:
    """HTML chips for every supported traffic class."""
    return "".join(
        tag(name, CLASS_TAG_COLORS.get(name, "cyan")) for name in CLASS_LABELS
    )
