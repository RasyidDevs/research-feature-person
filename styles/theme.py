"""Inject custom CSS into the Streamlit app — dark theme + responsive card grid."""

import streamlit as st

CSS = """
<style>
/* ── Import fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ───────────────────────────────────────────── */
:root {
    --bg:        #0e1117;
    --surface:   #161b22;
    --surface-2: #1c2333;
    --border:    #2a3140;
    --text:      #e6edf3;
    --text-dim:  #8b949e;
    --accent:    #58a6ff;
    --accent-bg: rgba(88,166,255,0.08);
    --green:     #3fb950;
    --orange:    #f0883e;
    --radius:    14px;
}

/* ── Force dark background everywhere ─────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.main, .block-container,
[data-testid="stMain"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stHeader"] {
    background-color: var(--bg) !important;
}

/* hide default streamlit menu & footer */
#MainMenu, footer, header {visibility: hidden;}

/* ── Title area ───────────────────────────────────────────────── */
.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0.15rem;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    font-size: 1.05rem;
    color: var(--text-dim);
    margin-top: 0;
}
.divider {
    height: 1px;
    background: var(--border);
    margin: 1.2rem 0 1.8rem;
}

/* ── Upload section ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    transition: border-color 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
}

.empty-hint {
    text-align: center;
    color: var(--text-dim);
    font-size: 0.92rem;
    margin-top: 1.5rem;
}
.file-count {
    font-size: 0.92rem;
    color: var(--text-dim);
    margin-top: 0.5rem;
}

/* ── Section title ────────────────────────────────────────────── */
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Responsive card grid ─────────────────────────────────────── */
.card-grid {
    display: grid;
    gap: 1.2rem;
    grid-template-columns: repeat(4, 1fr);   /* desktop: 4 per row */
}

@media (max-width: 1024px) {
    .card-grid {
        grid-template-columns: repeat(2, 1fr); /* tablet: 2 per row */
    }
}

@media (max-width: 640px) {
    .card-grid {
        grid-template-columns: 1fr;           /* mobile: 1 per row */
    }
}

/* ── Card ─────────────────────────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.85rem;
    transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
    display: flex;
    flex-direction: column;
}
.card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent), 0 8px 24px rgba(0,0,0,0.35);
    transform: translateY(-2px);
}

.card-img {
    width: 100%;
    border-radius: 10px;
    object-fit: contain;
    max-height: 240px;
    background: #0d1117;
}

.card-filename {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    margin: 0.55rem 0 0.6rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.card-label {
    font-weight: 600;
    font-size: 0.85rem;
    margin: 0.65rem 0 0.3rem;
    color: var(--text);
}

/* ── Tags (face accessories) ──────────────────────────────────── */
.tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.tag {
    display: inline-block;
    background: var(--accent-bg);
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.22rem 0.65rem;
    border-radius: 20px;
    border: 1px solid rgba(88,166,255,0.2);
}
.tag-empty {
    font-size: 0.82rem;
    color: var(--text-dim);
    font-style: italic;
}

/* ── Body badge ───────────────────────────────────────────────── */
.body-badge {
    display: inline-block;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    color: var(--badge-color, var(--accent));
    background: color-mix(in srgb, var(--badge-color, var(--accent)) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--badge-color, var(--accent)) 30%, transparent);
}

/* ── "Describe!" button ───────────────────────────────────────── */
.describe-btn-wrap {
    text-align: center;
    margin: 1.5rem 0;
}
div[data-testid="stButton"] > button.describe-btn,
.describe-btn-wrap + div [data-testid="stButton"] > button {
    background: linear-gradient(135deg, #58a6ff, #a78bfa) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.4rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: transform 0.2s, box-shadow 0.25s !important;
    letter-spacing: 0.3px;
}
div[data-testid="stButton"] > button.describe-btn:hover,
.describe-btn-wrap + div [data-testid="stButton"] > button:hover {
    transform: scale(1.04);
    box-shadow: 0 4px 20px rgba(88,166,255,0.35) !important;
}

/* ── Streamlit image override — do NOT fill container ─────────── */
[data-testid="stImage"] img {
    border-radius: 10px;
    object-fit: contain;
    max-height: 240px;
}

/* ── Spinner / progress dark friendly ─────────────────────────── */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
}

/* ── Sidebar polish ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
}

/* ── Status message cards ─────────────────────────────────────── */
.status-processing {
    text-align: center;
    padding: 1.5rem;
    color: var(--text-dim);
    font-size: 0.9rem;
}

.inference-spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 0.5rem;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)
