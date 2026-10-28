"""HTML/CSS building blocks for the Streamlit dashboard.

All dynamic text is HTML-escaped.  HTML strings are built without blank lines or
leading indentation so Markdown never mistakes them for code blocks.
"""
from __future__ import annotations

from html import escape

import config as cfg

P = cfg.PALETTE

def _theme_css(theme: str) -> str:
    """Theme overrides for app-authored components and Streamlit surfaces."""
    dark = """
    :root { --ink:#F1F5F9; --muted:#B6C2D1; --line:#334155; --primary:#60A5FA; --card:#172033; --page:#0B1120; --surface:#111827; --body:#E2E8F0; --soft:#1E293B;
      --btn-primary-bg:#2563EB; --btn-primary-fg:#FFFFFF; --btn-primary-hover-bg:#1D4ED8; --btn-primary-hover-fg:#FFFFFF;
      --btn-secondary-bg:#243247; --btn-secondary-fg:#F8FAFC; --btn-secondary-hover-bg:#334155; --btn-secondary-hover-fg:#FFFFFF;
      --btn-white-bg:#FFFFFF; --btn-white-fg:#0F172A; --btn-white-hover-bg:#E2E8F0; --btn-white-hover-fg:#0F172A;
      --btn-outline-bg:transparent; --btn-outline-fg:#BFDBFE; --btn-outline-hover-bg:#1E3A5F; --btn-outline-hover-fg:#FFFFFF;
      --control-bg:#172033; --control-fg:#F8FAFC; --control-border:#475569; }
    .stApp, [data-testid="stAppViewContainer"] { background:var(--page); color:var(--body); }
    [data-testid="stHeader"] { background:rgba(11,17,32,.94); }
    [data-testid="stSidebar"] { background:#0F172A; }
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] { border-color:var(--line)!important; }
    .gp-callout,.gp-step,.gp-status { background:var(--surface); color:var(--body); }
    .gp-callout .b,.gp-step .b { color:#CBD5E1; }
    .gp-flow .node,.gp-step .n { background:#1E3A5F; color:#DBEAFE; }
    .gp-kpi { background:var(--card); }
    .gp-kpi .value { color:#F8FAFC!important; }
    .stMarkdown, label, [data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] p { color:var(--body); }
    input, textarea, [data-baseweb="select"] > div, [data-baseweb="popover"], [role="listbox"], [role="option"] { background:#172033!important; color:#F8FAFC!important; }
    button, [data-testid="stDownloadButton"] button { border-color:#475569!important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border-color:var(--line); background:#111827!important; color:#E2E8F0!important; }
    [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:10px; }
    [data-testid="stAlert"] { background:#172033; color:#F8FAFC; }
    .gp-kpi .sub { color:#CBD5E1; }
    .gp-section .c { color:#B6C2D1; }
    """
    light = """
    :root { --ink:#0F172A; --muted:#64748B; --line:#E2E8F0; --primary:#2563EB; --card:#FFFFFF; --page:#F6F8FB; --surface:#FFFFFF; --body:#1E293B; --soft:#F1F5F9;
      --btn-primary-bg:#1D4ED8; --btn-primary-fg:#FFFFFF; --btn-primary-hover-bg:#1E40AF; --btn-primary-hover-fg:#FFFFFF;
      --btn-secondary-bg:#E2E8F0; --btn-secondary-fg:#0F172A; --btn-secondary-hover-bg:#CBD5E1; --btn-secondary-hover-fg:#0F172A;
      --btn-white-bg:#FFFFFF; --btn-white-fg:#0F172A; --btn-white-hover-bg:#E2E8F0; --btn-white-hover-fg:#0F172A;
      --btn-outline-bg:#FFFFFF; --btn-outline-fg:#1D4ED8; --btn-outline-hover-bg:#DBEAFE; --btn-outline-hover-fg:#1E40AF;
      --control-bg:#FFFFFF; --control-fg:#0F172A; --control-border:#CBD5E1; }
    .stApp, [data-testid="stAppViewContainer"] { background:var(--page); color:var(--body); }
    [data-testid="stSidebar"] { background:#FFFFFF; }
    .gp-callout,.gp-step { background:#FFFFFF; }
    .gp-status { background:#F1F5F9; color:#334155; }
    """
    if theme == "Dark":
        return dark
    if theme == "Light":
        return light
    return "@media (prefers-color-scheme: dark) {" + dark + "} @media (prefers-color-scheme: light) {" + light + "}"


CSS = f"""
<style>
:root {{ --ink:{P['ink']}; --muted:{P['muted']}; --line:{P['line']}; --primary:{P['primary']}; --card:{P['card']}; --page:{P['bg']}; --surface:#FFFFFF; --body:{P['ink']}; --soft:{P['soft_blue']}; }}

/* Component-specific accessible theme tokens. Never inherit a single global button foreground. */
.stApp {{ color:var(--body); }}

/* Streamlit expanders (including Advanced settings and Recent predictions).
   Their header is a button, so give it explicit theme-aware colors in every state. */
[data-testid="stExpander"] > details > summary,
[data-testid="stExpander"] > details > summary button,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary button {{
  background:var(--surface)!important;
  color:var(--body)!important;
  border-color:var(--line)!important;
}}
[data-testid="stExpander"] > details > summary:hover,
[data-testid="stExpander"] > details > summary:hover button,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:hover button {{
  background:var(--soft)!important;
  color:var(--ink)!important;
}}
[data-testid="stExpander"] summary:focus-visible,
[data-testid="stExpander"] summary button:focus-visible {{
  outline:2px solid var(--primary)!important;
  outline-offset:2px!important;
}}
[data-testid="stExpander"] > details > summary svg,
[data-testid="stExpander"] summary button svg {{
  color:var(--body)!important;
  fill:var(--body)!important;
}}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
  background:var(--surface)!important;
  color:var(--body)!important;
}}

.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {{
  transition: background-color .15s ease, color .15s ease, border-color .15s ease, box-shadow .15s ease;
}}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {{
  background:var(--btn-primary-bg)!important; color:var(--btn-primary-fg)!important; border-color:var(--btn-primary-bg)!important;
}}
.stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {{
  background:var(--btn-primary-hover-bg)!important; color:var(--btn-primary-hover-fg)!important;
}}
.stButton button[kind="secondary"], .stDownloadButton button, [data-testid="stFormSubmitButton"] button[kind="secondary"] {{
  background:var(--btn-secondary-bg)!important; color:var(--btn-secondary-fg)!important; border-color:var(--control-border)!important;
}}
.stButton button[kind="secondary"]:hover, .stDownloadButton button:hover, [data-testid="stFormSubmitButton"] button[kind="secondary"]:hover {{
  background:var(--btn-secondary-hover-bg)!important; color:var(--btn-secondary-hover-fg)!important;
}}
/* Explicit variants for app-authored white, outlined, navigation and export controls. */
.gp-button-white {{ background:var(--btn-white-bg)!important; color:var(--btn-white-fg)!important; border:1px solid var(--control-border)!important; }}
.gp-button-white:hover {{ background:var(--btn-white-hover-bg)!important; color:var(--btn-white-hover-fg)!important; }}
.gp-button-outline {{ background:var(--btn-outline-bg)!important; color:var(--btn-outline-fg)!important; border:1px solid var(--control-border)!important; }}
.gp-button-outline:hover {{ background:var(--btn-outline-hover-bg)!important; color:var(--btn-outline-hover-fg)!important; }}
.gp-button-nav {{ background:var(--btn-secondary-bg)!important; color:var(--btn-secondary-fg)!important; }}
.gp-button-nav:hover {{ background:var(--btn-secondary-hover-bg)!important; color:var(--btn-secondary-hover-fg)!important; }}
.gp-button-export {{ background:var(--btn-secondary-bg)!important; color:var(--btn-secondary-fg)!important; border:1px solid var(--control-border)!important; }}
.gp-button-export:hover {{ background:var(--btn-secondary-hover-bg)!important; color:var(--btn-secondary-hover-fg)!important; }}
.stButton button:focus-visible, .stDownloadButton button:focus-visible,
[data-testid="stFormSubmitButton"] button:focus-visible, [role="combobox"]:focus-visible {{
  outline:2px solid var(--primary)!important; outline-offset:2px!important; box-shadow:0 0 0 3px color-mix(in srgb, var(--primary) 24%, transparent)!important;
}}
.stButton button:active, .stDownloadButton button:active {{ filter:brightness(.92); }}
.stButton button:disabled, .stDownloadButton button:disabled, [data-testid="stFormSubmitButton"] button:disabled {{
  background:var(--soft)!important; color:var(--muted)!important; border-color:var(--line)!important; opacity:1!important; cursor:not-allowed!important;
}}
input, textarea, [data-baseweb="select"] > div, [data-baseweb="popover"], [role="listbox"], [role="option"] {{
  background:var(--control-bg)!important; color:var(--control-fg)!important; border-color:var(--control-border)!important;
}}
[data-baseweb="select"] input, [data-baseweb="select"] span {{ color:var(--control-fg)!important; }}
[data-baseweb="popover"] [role="option"]:hover, [role="option"][aria-selected="true"] {{ background:var(--soft)!important; color:var(--ink)!important; }}
.block-container {{ padding-top: 2.8rem; padding-bottom: 3rem; max-width: 1320px; overflow: visible; }}
html, body, [class*="css"] {{ font-family: -apple-system, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif; }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; color: var(--ink); }}
.gp-hero {{ overflow: visible; box-sizing: border-box; background: linear-gradient(120deg, {P['primary_dark']} 0%, {P['primary']} 100%); color: #fff; border-radius: 16px; padding: 26px 30px; margin-bottom: 18px; }}
.gp-hero h1 {{ color:#fff; font-size: clamp(1.55rem, 2.5vw, 2rem); line-height:1.35; overflow:visible; overflow-wrap:anywhere; margin: 0 0 8px 0; padding: 2px 0 0; }}
.gp-hero p {{ color: #DBEAFE; margin: 0 0 14px 0; font-size: 1.02rem; max-width: 820px; }}
.gp-chip {{ display:inline-block; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28); color:#fff; border-radius: 999px; padding: 3px 12px; margin: 0 8px 4px 0; font-size: 0.8rem; }}
.gp-kpi {{ box-sizing:border-box; width:100%; height:100%; min-height:184px; display:flex; flex-direction:column; background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
.gp-kpi .label {{ color: var(--muted); font-size: 0.74rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }}
.gp-kpi .value {{ font-size: 1.85rem; font-weight: 750; line-height: 1.2; margin: 6px 0 4px 0; }}
.gp-kpi .sub {{ color: var(--muted); font-size: 0.82rem; line-height: 1.4; margin-top:auto; padding-top:6px; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }}
.gp-section {{ margin: 26px 0 10px 0; padding-bottom: 6px; border-bottom: 2px solid var(--line); }}
.gp-section .t {{ font-size: 1.18rem; font-weight: 700; color: var(--ink); }}
.gp-section .c {{ color: var(--muted); font-size: 0.88rem; margin-top: 2px; }}
.gp-callout {{ border-radius: 12px; padding: 12px 16px; margin: 8px 0; border: 1px solid var(--line); border-left-width: 5px; background: #fff; }}
.gp-callout .t {{ font-weight: 700; color: var(--ink); font-size: 0.95rem; }}
.gp-callout .b {{ color: #334155; font-size: 0.9rem; margin-top: 2px; line-height: 1.45; }}
.gp-callout .s {{ float: right; color: var(--muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; border: 1px solid var(--line); border-radius: 6px; padding: 1px 7px; }}
.gp-step {{ background:#fff; border:1px solid var(--line); border-radius: 14px; padding: 16px 18px; min-height: 120px; }}
.gp-step .n {{ display:inline-block; width: 26px; height: 26px; line-height: 26px; text-align:center; border-radius: 50%; background: {P['soft_blue']}; color: {P['primary_dark']}; font-weight: 700; font-size: 0.85rem; margin-bottom: 8px; }}
.gp-step .t {{ font-weight: 700; color: var(--ink); }}
.gp-step .b {{ color: var(--muted); font-size: 0.87rem; margin-top: 2px; line-height: 1.4; }}
.gp-flow {{ display:flex; flex-wrap:wrap; gap: 6px; align-items:center; margin: 10px 0 4px 0; }}
.gp-flow .node {{ background: {P['soft_blue']}; color: {P['primary_dark']}; border-radius: 8px; padding: 6px 12px; font-weight: 600; font-size: 0.85rem; }}
.gp-flow .arrow {{ color: var(--muted); font-weight: 700; }}
.gp-brand {{ font-weight: 800; font-size: 1.05rem; color: var(--ink); }}
.gp-brand-sub {{ color: var(--muted); font-size: 0.8rem; margin-bottom: 10px; }}
.gp-status {{ background:#F1F5F9; border-radius: 10px; padding: 10px 12px; font-size: 0.8rem; color:#334155; line-height: 1.5; }}
.gp-tag {{ display:inline-block; border-radius: 6px; padding: 1px 8px; font-size: 0.75rem; font-weight: 600; }}
{_theme_css("System")}
</style>
"""


def inject_css(theme: str = "System") -> str:
    """Return CSS with a selected theme; System follows the OS preference."""
    return CSS.replace(_theme_css("System"), _theme_css(theme))


def hero(title: str, subtitle: str, chips: list[str]) -> str:
    chip_html = "".join(f'<span class="gp-chip">{escape(c)}</span>' for c in chips)
    return f'<div class="gp-hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p>{chip_html}</div>'


def kpi(label: str, value: str, sub: str = "", color: str = P["ink"]) -> str:
    return (f'<div class="gp-kpi"><div class="label">{escape(label)}</div>'
            f'<div class="value" style="color:{color}">{escape(value)}</div><div class="sub">{escape(sub)}</div></div>')


def section(title: str, caption: str = "") -> str:
    cap = f'<div class="c">{escape(caption)}</div>' if caption else ""
    return f'<div class="gp-section"><div class="t">{escape(title)}</div>{cap}</div>'


_KIND_COLOR = {"success": P["good"], "info": P["primary"], "warning": P["warn"], "error": P["bad"]}


def callout(kind: str, title: str, body: str, source: str = "") -> str:
    color = _KIND_COLOR.get(kind, P["primary"])
    tag = f'<span class="s">{escape(source)}</span>' if source else ""
    return (f'<div class="gp-callout" style="border-left-color:{color}">{tag}'
            f'<div class="t">{escape(title)}</div><div class="b">{escape(body)}</div></div>')


def step(number: int, title: str, body: str) -> str:
    return (f'<div class="gp-step"><div class="n">{number}</div><div class="t">{escape(title)}</div>'
            f'<div class="b">{escape(body)}</div></div>')


def flow(nodes: list[str]) -> str:
    parts = '<span class="arrow">&rarr;</span>'.join(f'<span class="node">{escape(n)}</span>' for n in nodes)
    return f'<div class="gp-flow">{parts}</div>'


def sidebar_brand() -> str:
    return '<div class="gp-brand">Game Performance AI</div><div class="gp-brand-sub">Prediction and optimization</div>'


def status_box(lines: list[str]) -> str:
    return '<div class="gp-status">' + "<br>".join(escape(l) for l in lines) + "</div>"


def tag(text: str, color: str) -> str:
    return f'<span class="gp-tag" style="background:{color}22;color:{color}">{escape(text)}</span>'
