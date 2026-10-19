from __future__ import annotations
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import SAMPLE_PREVIEW_PATH, TARGET_COLUMN
from src.artifacts import ModelBundle
from src.data import FlowFileError, load_flow_file
from src.error_analysis import analyze_errors
from src.explainability import explain_prediction, shap_capable
from src.prediction import (
    PredictionInputError,
    predict_flows,
    recommendation_for,
    summarize_predictions,
)
from ui.components import (
    alert_card,
    metric_card,
    section_title,
    severity_badge,
)

SUPPORTED_FORMATS = ["csv", "xlsx", "xls", "json"]

_SAMPLE_SOURCE = "__sample__"

_STATE_KEYS = (
    "scan_source",
    "scan_bundle",
    "scan_results",
    "scan_probabilities",
    "scan_has_truth",
)


def render(bundle: ModelBundle) -> None:
    section_title(
        "Threat Scanner",
        "Upload recorded traffic (CSV, Excel, or JSON) and classify every flow.",
    )

    uploaded = st.file_uploader(
        "Traffic file",
        type=SUPPORTED_FORMATS,
        help="Flow records with the same feature columns as the training data.",
    )

    if uploaded is None:
        st.info(
            "Upload a file to begin. Required: the model's traffic feature columns "
            f"({len(bundle.feature_names)} columns). Optional: a `{TARGET_COLUMN}` "
            "column with ground-truth labels — when present, the scanner also "
            "computes a full evaluation (confusion matrix, FP/FN analysis)."
        )
        _render_sample_hint(bundle)
        _render_persisted(bundle, expected_source=_SAMPLE_SOURCE)
        return

    try:
        df = load_flow_file(io.BytesIO(uploaded.getvalue()), filename=uploaded.name)
    except FlowFileError as exc:
        st.error(str(exc))
        return

    missing = [c for c in bundle.feature_names if c not in df.columns]
    has_truth = TARGET_COLUMN in df.columns

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Records", f"{len(df):,}", uploaded.name, accent="cyan")
    with col2:
        metric_card(
            "Columns",
            str(df.shape[1]),
            f"{len(bundle.feature_names)} required features",
            accent="violet",
        )
    with col3:
        metric_card(
            "Missing values", f"{int(df.isna().sum().sum()):,}", "raw upload", accent="teal"
        )
    with col4:
        metric_card(
            "Ground truth",
            "Present" if has_truth else "Not provided",
            "enables evaluation" if has_truth else "predictions only",
            accent="amber",
        )

    with st.expander("Data preview", expanded=False):
        st.dataframe(df.head(20), width="stretch", hide_index=True)

    if missing:
        _clear_results()
        alert_card(
            "danger",
            f"<strong>Cannot classify:</strong> {len(missing)} required feature(s) "
            f"missing — <code>{', '.join(missing[:6])}"
            + (" ..." if len(missing) > 6 else "")
            + "</code>. The scanner will not invent traffic statistics; fix the "
            "input file and re-upload.",
        )
        st.caption(
            "Tip: `data/cicids2017_preview.csv` is a ready-made file in the "
            "training schema that works as-is."
        )
        return

    source_key = f"{uploaded.name}:{uploaded.size}"
    _reconcile_state(source_key, bundle)

    if st.button("Run classification", type="primary", key="run_scan"):
        _compute_and_store(bundle, df, has_truth, source_key)

    _render_persisted(bundle, expected_source=source_key)


# ---------------------------------------------------------------------------
# state management
# ---------------------------------------------------------------------------


def _clear_results() -> None:
    for key in _STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("record_select", None)


def _reconcile_state(source_key: str, bundle: ModelBundle) -> None:
    """Drop results that belong to a different file or a different model."""
    if (
        st.session_state.get("scan_source") not in (None, source_key)
        or st.session_state.get("scan_bundle") != bundle.model_name
    ):
        _clear_results()


def _compute_and_store(
    bundle: ModelBundle, df: pd.DataFrame, has_truth: bool, source_key: str
) -> None:
    """Run the (expensive) prediction once and cache it for later reruns."""
    with st.spinner("Classifying flows ..."):
        try:
            batch = predict_flows(bundle, df)
        except PredictionInputError as exc:
            st.error(str(exc))
            _clear_results()
            return
    st.session_state["scan_source"] = source_key
    st.session_state["scan_bundle"] = bundle.model_name
    st.session_state["scan_results"] = batch.results
    st.session_state["scan_probabilities"] = batch.probabilities
    st.session_state["scan_has_truth"] = has_truth and TARGET_COLUMN in df.columns


def _render_persisted(bundle: ModelBundle, expected_source: str) -> None:
    """Render cached results, if they belong to the file currently in view."""
    results = st.session_state.get("scan_results")
    if results is None or st.session_state.get("scan_source") != expected_source:
        return
    if st.session_state.get("scan_bundle") != bundle.model_name:
        return

    has_truth = bool(st.session_state.get("scan_has_truth")) and (
        TARGET_COLUMN in results.columns
    )

    _render_summary(bundle, summarize_predictions(results), results)
    _render_distributions(results)

    st.download_button(
        "Download classification results (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="nids_classification_results.csv",
        mime="text/csv",
        key="download_scan",
    )

    if has_truth:
        _render_evaluation(bundle, results)

    _render_record_explorer(bundle, results)


# ---------------------------------------------------------------------------
# result sections
# ---------------------------------------------------------------------------


def _render_summary(bundle, summary, results: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(
            "Total records", f"{summary.total_records:,}", "flows classified", accent="cyan"
        )
    with col2:
        metric_card(
            "Flagged as attack",
            f"{summary.attack_count:,}",
            f"{summary.attack_rate:.2f}% of traffic",
            accent="rose",
        )
    with col3:
        metric_card(
            "Benign",
            f"{summary.normal_count:,}",
            "classified as normal traffic",
            accent="teal",
        )
    with col4:
        metric_card(
            "Health verdict",
            summary.health,
            "based on the flagged share",
            accent="amber",
        )

    kind = (
        "danger"
        if summary.health == "High Risk"
        else ("warning" if summary.health in ("Medium Risk", "Low Risk") else "success")
    )
    alert_card(kind, f"<strong>{summary.health}.</strong> {summary.recommendation}")


def _render_distributions(results: pd.DataFrame) -> None:
    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Predictions by class**")
        st.bar_chart(results["Prediction"].value_counts(), height=300)
    with col2:
        st.markdown("**Severity distribution**")
        order = ["Low", "Medium", "High", "Critical"]
        severity = results["Severity"].value_counts().reindex(order).fillna(0).astype(int)
        st.bar_chart(severity, height=300)


def _render_evaluation(bundle: ModelBundle, results: pd.DataFrame) -> None:
    st.markdown("")
    section_title(
        "Ground-truth evaluation",
        "This upload carries labels, so predictions are scored against them.",
    )

    try:
        analysis = analyze_errors(results, bundle.classes, model_name=bundle.model_label)
    except KeyError as exc:
        st.warning(str(exc))
        return

    evaluation = analysis.evaluation
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Accuracy", f"{evaluation.accuracy:.2%}", "on this upload", accent="cyan")
    with col2:
        metric_card(
            "Macro F1",
            f"{evaluation.f1_macro:.4f}",
            "unweighted across classes",
            accent="violet",
        )
    with col3:
        metric_card(
            "Attack FPR",
            f"{evaluation.attack_fpr:.2%}",
            "benign flows flagged",
            accent="amber",
        )
    with col4:
        metric_card(
            "Errors",
            f"{analysis.fp_count + analysis.fn_count:,}",
            f"{analysis.fp_count} false alarms, {analysis.fn_count} misses",
            accent="rose",
        )

    st.markdown("")
    tab_confusion, tab_perclass, tab_errors = st.tabs(
        ["Confusion matrix", "Per-class metrics", "FP / FN analysis"]
    )

    with tab_confusion:
        st.pyplot(_confusion_figure(evaluation), width="stretch")
        st.dataframe(evaluation.confusion, width="stretch")

    with tab_perclass:
        st.dataframe(
            evaluation.per_class.round(4), width="stretch", hide_index=True
        )
        with st.expander("Sklearn classification report"):
            st.code(evaluation.report_text, language="text")

    with tab_errors:
        _render_error_tables(results, analysis)


def _render_error_tables(results: pd.DataFrame, analysis) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**False positives** — benign flows flagged as attacks ({analysis.fp_count})")
        if analysis.false_positives.empty:
            st.success("No benign flows were flagged.")
        else:
            st.dataframe(
                analysis.false_positives.head(50), width="stretch", hide_index=True
            )
            st.download_button(
                "Download FP table",
                analysis.false_positives.to_csv(index=False).encode("utf-8"),
                file_name="false_positives.csv",
                mime="text/csv",
                key="dl_fp",
            )
    with col2:
        st.markdown(f"**Missed / mislabelled attacks** ({analysis.fn_count})")
        if analysis.false_negatives.empty:
            st.success("No attack flows were missed.")
        else:
            st.dataframe(
                analysis.false_negatives.head(50), width="stretch", hide_index=True
            )
            st.download_button(
                "Download FN table",
                analysis.false_negatives.to_csv(index=False).encode("utf-8"),
                file_name="false_negatives.csv",
                mime="text/csv",
                key="dl_fn",
            )

    if not analysis.top_confusions.empty:
        st.markdown("**Dominant confusions**")
        st.dataframe(analysis.top_confusions, width="stretch", hide_index=True)

    if "Confidence" in results.columns and (analysis.fp_count + analysis.fn_count):
        st.caption(
            "Mistakes are sorted by model confidence — high-confidence errors are "
            "systematic (worth investigating first); low-confidence ones are "
            "borderline cases."
        )


def _render_record_explorer(bundle: ModelBundle, results: pd.DataFrame) -> None:
    st.markdown("")
    section_title(
        "Record explorer",
        "Inspect any classified flow, including why the model decided it.",
    )

    index = st.selectbox(
        "Record",
        options=list(results.index),
        format_func=lambda i: (
            f"#{i} — {results.loc[i, 'Prediction']}"
        ),
        key="record_select",
    )
    row = results.loc[[index]]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"<div class='result-title'>{row['Prediction'].iloc[0]}</div>"
            + severity_badge(str(row["Severity"].iloc[0])),
            unsafe_allow_html=True,
        )
        if "Confidence" in row.columns:
            st.markdown(
                f"<div class='confidence-value'>{row['Confidence'].iloc[0]:.1%}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div class='recommendation-box'>"
            f"{recommendation_for(str(row['Severity'].iloc[0]))}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Input features**")
        st.dataframe(
            row[bundle.feature_names].T.rename(columns={index: "value"}),
            width="stretch",
        )
    with col2:
        _render_flow_explanation(bundle, row, index)


def _render_flow_explanation(bundle: ModelBundle, row: pd.DataFrame, index) -> None:
    if not shap_capable(bundle):
        st.info(
            "Per-flow SHAP explanations require the XGBoost production model; "
            "this bundle is not tree-based."
        )
        return

    features = row[bundle.feature_names]
    explanation = explain_prediction(bundle, features, top_n=12)
    if explanation is None:
        st.info("No explanation available for this flow.")
        return

    st.markdown(f"**Why this decision** — predicted `{explanation.predicted_class}`")
    st.caption(
        "SHAP values (margin space): positive bars push toward the predicted "
        "class, negative bars push away."
    )
    contrib = explanation.contributions.sort_values("Contribution")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#0d9488" if v >= 0 else "#e11d48" for v in contrib["Contribution"]]
    ax.barh(contrib["Feature"], contrib["Contribution"], color=colors)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel("SHAP contribution")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.caption(
        f"Base value {explanation.base_value:.3f} + contributions "
        f"= model score {explanation.total:.3f}."
    )


def _confusion_figure(evaluation) -> "plt.Figure":
    matrix = evaluation.confusion.to_numpy()
    labels = [c.replace("true: ", "") for c in evaluation.confusion.index]

    fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (upload)", fontsize=11)

    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "#0f172a",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _render_sample_hint(bundle: ModelBundle) -> None:
    """Offer the bundled sample file when the user has nothing to upload."""
    if not SAMPLE_PREVIEW_PATH.exists():
        return

    with st.expander("Don't have a traffic file? Try the bundled sample"):
        st.caption(
            f"`{SAMPLE_PREVIEW_PATH.name}` contains multi-class sample rows in the "
            "exact training schema."
        )
        if st.button("Load bundled sample", key="load_sample"):
            pending = pd.read_csv(SAMPLE_PREVIEW_PATH)
            st.session_state["sample_to_scan"] = pending
            st.rerun()

    pending = st.session_state.pop("sample_to_scan", None)
    if pending is not None:
        _compute_and_store(
            bundle,
            pending,
            has_truth=TARGET_COLUMN in pending.columns,
            source_key=_SAMPLE_SOURCE,
        )
