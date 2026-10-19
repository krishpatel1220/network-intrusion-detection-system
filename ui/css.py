from __future__ import annotations
import streamlit as st

CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

/* =========================================================
   DESIGN TOKENS
========================================================= */

:root {
    --c-bg: #f7f9fc;
    --c-surface: #ffffff;
    --c-border: #e2e8f0;
    --c-text: #1e293b;
    --c-text-strong: #0f172a;
    --c-text-muted: #64748b;

    --c-cyan: #0891b2;      --c-cyan-light: #ecfeff;    --c-cyan-border: #a5f3fc;
    --c-violet: #7c3aed;    --c-violet-light: #f5f3ff;  --c-violet-border: #ddd6fe;
    --c-amber: #d97706;     --c-amber-light: #fffbeb;   --c-amber-border: #fde68a;
    --c-teal: #0d9488;      --c-teal-light: #f0fdfa;    --c-teal-border: #99f6e4;
    --c-rose: #e11d48;      --c-rose-light: #fff1f2;    --c-rose-border: #fecdd3;

    --sev-low: #16a34a;
    --sev-medium: #d97706;
    --sev-high: #ea580c;
    --sev-critical: #dc2626;

    --sidebar-bg: #0b1220;
    --sidebar-bg-raised: #131c2e;
    --sidebar-border: #1f2a3d;
    --sidebar-text: #cbd5e1;
    --sidebar-text-muted: #7c8aa5;

    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --ease: cubic-bezier(0.4, 0, 0.2, 1);
}

/* =========================================================
   GLOBAL
========================================================= */

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1220px;
}

body {
    background-color: var(--c-bg);
    color: var(--c-text);
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
}

.stApp { background: var(--c-bg); color: var(--c-text); }

h1, h2, h3, h4, h5, h6 {
    color: var(--c-text-strong) !important;
    font-weight: 700;
    letter-spacing: -0.01em;
}

p, label { color: var(--c-text); }

code, .mono {
    font-family: "JetBrains Mono", ui-monospace, monospace !important;
}


/* =========================================================
   SECTION HEADERS
========================================================= */

.section-header {
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--c-text-strong);
    margin-top: 0.6rem;
    margin-bottom: 0.3rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--c-border);
    position: relative;
}

.section-header::after {
    content: "";
    position: absolute;
    bottom: -2px; left: 0;
    width: 46px; height: 2px;
    background: linear-gradient(90deg, var(--c-cyan), var(--c-violet));
    border-radius: 999px;
}

.subtle {
    color: var(--c-text-muted);
    font-size: 0.95rem;
    margin-bottom: 0.7rem;
    line-height: 1.5;
}


/* =========================================================
   METRIC CARDS
========================================================= */

.metric-card {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius-lg);
    border-top: 3px solid var(--c-border);
    padding: 1.05rem 1.2rem;
    height: 100%;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease);
}

.metric-card:hover {
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10);
    transform: translateY(-2px);
}

.metric-card.accent-cyan   { border-top-color: var(--c-cyan); }
.metric-card.accent-violet { border-top-color: var(--c-violet); }
.metric-card.accent-amber  { border-top-color: var(--c-amber); }
.metric-card.accent-teal   { border-top-color: var(--c-teal); }
.metric-card.accent-rose   { border-top-color: var(--c-rose); }

.metric-title {
    font-size: 0.72rem;
    color: var(--c-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
    font-weight: 700;
}

.metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--c-text-strong);
    word-break: break-word;
    line-height: 1.2;
    font-family: "JetBrains Mono", ui-monospace, monospace;
}

.metric-caption {
    font-size: 0.82rem;
    color: var(--c-text-muted);
    margin-top: 0.45rem;
    line-height: 1.35;
}


/* =========================================================
   SEVERITY BADGES
========================================================= */

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: 0.02em;
}

.badge::before {
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    background: rgba(255,255,255,0.85);
}

.badge-low      { background: var(--sev-low); }
.badge-medium   { background: var(--sev-medium); }
.badge-high     { background: var(--sev-high); }
.badge-critical { background: var(--sev-critical); }

/* Generic colored chip for attack-type tags */
.tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.83rem;
    font-weight: 600;
    margin: 0.2rem 0.3rem 0.2rem 0;
    border: 1px solid transparent;
}

.tag-cyan   { background: var(--c-cyan-light);   color: var(--c-cyan);   border-color: var(--c-cyan-border); }
.tag-violet { background: var(--c-violet-light); color: var(--c-violet); border-color: var(--c-violet-border); }
.tag-amber  { background: var(--c-amber-light);  color: var(--c-amber);  border-color: var(--c-amber-border); }
.tag-teal   { background: var(--c-teal-light);   color: var(--c-teal);   border-color: var(--c-teal-border); }
.tag-rose   { background: var(--c-rose-light);   color: var(--c-rose);   border-color: var(--c-rose-border); }


/* =========================================================
   BUTTONS
========================================================= */

[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button,
[data-testid="stFormSubmitButton"] button {
    border-radius: var(--radius-sm);
    border: 1px solid #d1d5db;
    background: var(--c-surface);
    color: var(--c-text) !important;
    font-weight: 600;
    padding: 0.55rem 1rem;
    transition: all 0.2s var(--ease);
}

[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    border-color: var(--c-cyan);
    color: var(--c-cyan) !important;
    background: var(--c-cyan-light);
    box-shadow: 0 4px 12px rgba(8, 145, 178, 0.14);
}

[data-testid="stButton"] button[kind="primary"],
[data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--c-cyan), #0e7490) !important;
    border-color: var(--c-cyan) !important;
    color: #ffffff !important;
    box-shadow: 0 6px 18px rgba(8, 145, 178, 0.25);
}

[data-testid="stButton"] button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0e7490, #155e75) !important;
    border-color: #0e7490 !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}


/* =========================================================
   INPUTS
========================================================= */

div[data-baseweb="select"] > div {
    background: var(--c-surface) !important;
    border: 1px solid #d1d5db !important;
    border-radius: var(--radius-sm) !important;
}

div[data-baseweb="select"] > div:hover { border-color: var(--c-cyan) !important; }

input, textarea {
    background-color: var(--c-surface) !important;
    color: var(--c-text) !important;
    border-radius: var(--radius-sm) !important;
}

input:focus, textarea:focus {
    border-color: var(--c-cyan) !important;
    box-shadow: 0 0 0 2px var(--c-cyan-light) !important;
}

/* Radio pills used inside forms */
div[role="radiogroup"] label {
    border: 1px solid var(--c-border);
    background: var(--c-surface);
    border-radius: 999px;
    padding: 0.3rem 0.85rem !important;
    margin-right: 0.4rem;
    transition: all 0.2s var(--ease);
}

div[role="radiogroup"] label:hover {
    border-color: var(--c-cyan);
    background: var(--c-cyan-light);
}


/* =========================================================
   FILE UPLOADER
========================================================= */

[data-testid="stFileUploaderDropzone"] {
    background: var(--c-bg);
    border: 1.5px dashed #cbd5e1;
    border-radius: var(--radius-lg);
    padding: 1rem;
    transition: all 0.2s var(--ease);
}

[data-testid="stFileUploaderDropzone"]:hover {
    background: var(--c-cyan-light);
    border-color: var(--c-cyan);
}

[data-testid="stFileUploaderDropzone"] * { color: var(--c-text) !important; }
[data-testid="stFileUploaderDropzone"] svg { fill: var(--c-cyan) !important; }


/* =========================================================
   SIDEBAR (dark "operations center" theme)
========================================================= */

section[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid var(--sidebar-border);
}

section[data-testid="stSidebar"] * { color: var(--sidebar-text); }

section[data-testid="stSidebar"] hr {
    border-top: 1px solid var(--sidebar-border);
    margin: 1rem 0;
}

section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.35rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex;
    align-items: center;
    width: 100%;
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    padding: 0.6rem 0.8rem !important;
    margin-right: 0;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.18s var(--ease);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: var(--sidebar-bg-raised);
    border-color: var(--sidebar-border);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(8,145,178,0.22), rgba(124,58,237,0.16));
    border-color: var(--c-cyan);
    box-shadow: inset 0 0 0 1px rgba(8,145,178,0.25);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
}

.sidebar-brand {
    font-size: 1.2rem;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 0.1rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

.sidebar-tagline {
    font-size: 0.78rem;
    color: var(--sidebar-text-muted) !important;
    margin-bottom: 0.85rem;
    line-height: 1.4;
}

.sb-chip-row {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}

.sb-chip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--sidebar-bg-raised);
    border: 1px solid var(--sidebar-border);
    border-left: 3px solid var(--c-cyan);
    border-radius: 8px;
    padding: 0.4rem 0.65rem;
    font-size: 0.78rem;
}

.sb-chip.chip-cyan   { border-left-color: var(--c-cyan); }
.sb-chip.chip-violet { border-left-color: var(--c-violet); }
.sb-chip.chip-amber  { border-left-color: var(--c-amber); }

.sb-chip-label {
    color: var(--sidebar-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.66rem;
    font-weight: 700;
}

.sb-chip-value {
    color: #ffffff;
    font-weight: 700;
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 0.78rem;
}


/* =========================================================
   ALERT / INFO CARDS
========================================================= */

.info-card {
    background: var(--c-cyan-light); border: 1px solid var(--c-cyan-border);
    border-left: 4px solid var(--c-cyan); border-radius: 8px;
    padding: 0.85rem 1rem; margin: 0.7rem 0; color: #155e75;
}

.success-card {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid var(--sev-low); border-radius: 8px;
    padding: 0.85rem 1rem; margin: 0.7rem 0; color: #166534;
}

.warning-card {
    background: var(--c-amber-light); border: 1px solid var(--c-amber-border);
    border-left: 4px solid var(--c-amber); border-radius: 8px;
    padding: 0.85rem 1rem; margin: 0.7rem 0; color: #92400e;
}

.danger-card {
    background: var(--c-rose-light); border: 1px solid var(--c-rose-border);
    border-left: 4px solid var(--c-rose); border-radius: 8px;
    padding: 0.85rem 1rem; margin: 0.7rem 0; color: #9f1239;
}


/* =========================================================
   DATAFRAME / TABLE
========================================================= */

[data-testid="stDataFrame"] {
    border: 1px solid var(--c-border);
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
}


/* =========================================================
   TABS
========================================================= */

button[data-baseweb="tab"] { color: var(--c-text-muted) !important; font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--c-cyan) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--c-cyan) !important; }


/* =========================================================
   EXPANDERS
========================================================= */

[data-testid="stExpander"] {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--radius-md);
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
}


/* =========================================================
   FORM CARDS (manual prediction)
========================================================= */

.slider-card-title {
    font-weight: 700;
    color: var(--c-text-strong);
    font-size: 0.95rem;
    margin-bottom: 0.15rem;
}

.slider-card-desc {
    color: var(--c-text-muted);
    font-size: 0.8rem;
    margin-bottom: 0.55rem;
    line-height: 1.4;
}

.result-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--c-text-strong);
    margin-bottom: 0.35rem;
}

.confidence-value {
    font-size: 1.6rem;
    font-weight: 800;
    font-family: "JetBrains Mono", ui-monospace, monospace;
    color: var(--c-cyan);
    margin: 0.2rem 0 0.5rem 0;
}

.recommendation-box {
    background: var(--c-bg);
    border: 1px solid var(--c-border);
    border-radius: var(--radius-sm);
    padding: 0.8rem 1rem;
    margin-top: 0.9rem;
    font-size: 0.92rem;
    line-height: 1.5;
}


/* =========================================================
   DIVIDERS / FOOTER
========================================================= */

hr { border: none; border-top: 1px solid var(--c-border); margin: 1.3rem 0; }

.footer {
    text-align: center;
    padding: 30px 0 15px 0;
    margin-top: 50px;
    border-top: 1px solid var(--c-border);
    color: #9ca3af;
    font-size: 13px;
}

.footer p { margin-bottom: 6px; color: #4b5563; font-size: 14px; font-weight: 600; }
.footer span { color: #9ca3af; }

</style>
"""


def load_css() -> None:
    """Inject the dashboard stylesheet."""
    st.markdown(CSS, unsafe_allow_html=True)
