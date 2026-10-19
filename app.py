from __future__ import annotations

import streamlit as st

from config import PRODUCTION_MODEL_NAME
from ui import css
from ui.components import load_production_bundle
from ui.pages import dataset, explainability, manual, overview, performance, scanner

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Dashboard": overview,
    "Threat Scanner": scanner,
    "Manual Prediction": manual,
    "Model Performance": performance,
    "Explainability": explainability,
    "Dataset Insights": dataset,
}
# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar(bundle) -> str:
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-brand'>🛡 NIDS</div>"
            "<div class='sidebar-tagline'>AI-powered threat detection & "
            "traffic classification</div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            list(PAGES),
            key="nav_page",
            label_visibility="collapsed",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        if bundle is not None:
            metrics = bundle.metrics.get("test", {})
            accuracy = metrics.get("accuracy")

            st.markdown(
                "<div class='sb-chip-row'>"
                + _chip("Model", bundle.model_label, "cyan")
                + _chip(
                    "Test accuracy",
                    f"{accuracy:.2%}" if accuracy is not None else "n/a",
                    "violet",
                )
                + _chip(
                    "Features",
                    str(len(bundle.feature_names)),
                    "amber",
                )
                + _chip(
                    "Classes",
                    str(len(bundle.classes)),
                    "cyan",
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown(
            "<div class='sidebar-tagline'>"
            "CICIDS2017 · XGBoost + scikit-learn"
            "</div>",
            unsafe_allow_html=True,
        )

    return page


# ---------------------------------------------------------------------------
# Sidebar metric chip
# ---------------------------------------------------------------------------

def _chip(label: str, value: str, color: str) -> str:
    return (
        f"<div class='sb-chip chip-{color}'>"
        f"<span class='sb-chip-label'>{label}</span>"
        f"<span class='sb-chip-value'>{value}</span>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def _footer() -> None:
    st.markdown(
        "<div class='footer'>"
        "<p>Network Intrusion Detection System</p>"
        "<span>CICIDS2017-based traffic classification · "
        "refactored, tested pipeline with SHAP explainability</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    css.load_css()

    try:
        bundle = load_production_bundle(PRODUCTION_MODEL_NAME)

    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Quick start:\n\n"
            "```\n"
            "pip install -r requirements.txt\n"
            "python prepare_data.py   # needs raw CICIDS2017 files in data/raw/\n"
            "python train.py\n"
            "streamlit run app.py\n"
            "```"
        )
        st.stop()

    except Exception as exc:
        st.error(f"Could not load the model bundle: {exc}")
        st.info("Re-generate it with `python train.py`.")
        st.stop()

    page = _sidebar(bundle)

    try:
        PAGES[page].render(bundle)
    except Exception as exc:
        st.exception(exc)

    _footer()


if __name__ == "__main__":
    main()