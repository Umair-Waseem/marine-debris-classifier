import html
from typing import List, Optional

import streamlit as st


def apply_project_style():
    """Apply shared visual styling for the Streamlit demo apps."""
    st.markdown(
        """
        <style>
        :root {
            --ui-primary: #0f766e;
            --ui-primary-soft: #e6fffb;
            --ui-ink: #102027;
            --ui-muted: #5f6f78;
            --ui-border: #d9e2e7;
            --ui-panel: #ffffff;
            --ui-panel-soft: #f7faf9;
            --ui-accent: #b7791f;
        }

        .main .block-container {
            max-width: 1160px;
            padding-top: 2rem;
            padding-bottom: 2.8rem;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        div[data-testid="stSidebar"] {
            border-right: 1px solid var(--ui-border);
        }

        div[data-testid="stSidebar"] h1,
        div[data-testid="stSidebar"] h2,
        div[data-testid="stSidebar"] h3 {
            color: var(--ui-ink);
        }

        .hero-panel {
            border: 1px solid var(--ui-border);
            border-left: 6px solid var(--ui-primary);
            background: linear-gradient(90deg, #ffffff 0%, #f2fbf9 100%);
            border-radius: 10px;
            padding: 1.35rem 1.5rem 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }

        .hero-eyebrow {
            color: var(--ui-primary);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            color: var(--ui-ink);
            font-size: 1.95rem;
            line-height: 1.15;
            font-weight: 750;
            margin: 0;
        }

        .hero-subtitle {
            color: var(--ui-muted);
            font-size: 0.98rem;
            line-height: 1.5;
            margin: 0.6rem 0 0 0;
            max-width: 800px;
        }

        .tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 1rem;
        }

        .tag {
            border: 1px solid #b7ded8;
            background: var(--ui-primary-soft);
            color: #115e59;
            border-radius: 999px;
            padding: 0.22rem 0.62rem;
            font-size: 0.76rem;
            font-weight: 650;
        }

        .section-heading {
            margin-top: 1rem;
            margin-bottom: 0.45rem;
        }

        .section-heading h2 {
            font-size: 1.22rem;
            margin-bottom: 0.18rem;
        }

        .section-heading p {
            margin: 0;
            color: var(--ui-muted);
            line-height: 1.45;
        }

        .note-panel {
            border: 1px solid #f1d594;
            background: #fffaf0;
            color: #5f4312;
            border-radius: 8px;
            padding: 0.78rem 0.9rem;
            line-height: 1.45;
            margin: 0.5rem 0 0.9rem 0;
        }

        .hint-text {
            color: var(--ui-muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin: 0.35rem 0 0.65rem 0;
        }

        .result-banner {
            border: 1px solid #9bd3ca;
            background: #effdfa;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            margin: 0.45rem 0 0.8rem 0;
        }

        .result-label {
            color: #11645d;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .result-value {
            color: var(--ui-ink);
            font-size: 1.35rem;
            line-height: 1.2;
            font-weight: 780;
            margin: 0;
        }

        .subtle-panel {
            border: 1px solid var(--ui-border);
            background: var(--ui-panel-soft);
            border-radius: 10px;
            padding: 1rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid var(--ui-border);
            border-radius: 10px;
            padding: 0.75rem 0.85rem;
            min-height: 88px;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--ui-muted);
            font-weight: 650;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ui-ink);
            font-weight: 760;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid #0f766e;
            font-weight: 650;
        }

        .stButton > button[kind="primary"] {
            background: #0f766e;
            border-color: #0f766e;
        }

        div[data-testid="stFileUploader"] section {
            border-color: var(--ui-border);
            border-radius: 10px;
            background: #fbfdfc;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        div[data-testid="stTabs"] button {
            font-weight: 650;
        }

        .footer-note {
            color: var(--ui-muted);
            font-size: 0.86rem;
            text-align: center;
            padding-top: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(
    title: str,
    subtitle: str,
    eyebrow: str,
    tags: Optional[List[str]] = None,
):
    """Render a consistent academic-demo hero block."""
    tags = tags or []
    tag_html = "".join(
        f'<span class="tag">{html.escape(tag)}</span>'
        for tag in tags
    )

    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-eyebrow">{html.escape(eyebrow)}</div>
            <h1 class="hero-title">{html.escape(title)}</h1>
            <p class="hero-subtitle">{html.escape(subtitle)}</p>
            <div class="tag-row">{tag_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, body: Optional[str] = None):
    """Render a compact section heading with optional support text."""
    body_html = f"<p>{html.escape(body)}</p>" if body else ""

    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{html.escape(title)}</h2>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_note(text: str):
    """Render a restrained warning or limitation note."""
    st.markdown(
        f'<div class="note-panel">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_hint(text: str):
    """Render concise supporting text."""
    st.markdown(
        f'<div class="hint-text">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_result_banner(label: str, value: str):
    """Render a compact highlighted result."""
    st.markdown(
        f"""
        <div class="result-banner">
            <div class="result-label">{html.escape(label)}</div>
            <p class="result-value">{html.escape(value)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(text: str):
    """Render a consistent footer."""
    st.markdown(
        f'<div class="footer-note">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )
