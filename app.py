"""Streamlit UI for the 100% local Codebase QA engine.

Pipeline behind the interface:
    GitHub repo -> AST parse + symbol graph (Phase 1)
                -> hybrid retrieval: BM25 + jina dense + RRF + cross-encoder
                   + depth-1 call-graph expansion (Phase 2)
                -> local Ollama model generates the answer (Phase 3)

Design language: dark slate (#0B0F19) with a faint grid + radial glow, layered
glassmorphism panels (blur + 1px translucent borders + depth shadow), Geist/Inter
type, a sky/teal accent. Reference points: Linear, actionai.co.
Everything runs locally: no API keys, no cloud calls.
"""

from __future__ import annotations

import glob
import html
import json
import os
import re
import sqlite3

import streamlit as st
import streamlit.components.v1 as components

from indexer import GRAPH_DIR, index_repo
from querier import answer_question, ollama_models

st.set_page_config(page_title="Codebase QA", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")

ROLE_COLOR = {
    "seed": "#e8590c",       # the re-ranked hit
    "callee": "#1c7ed6",     # functions it calls
    "caller": "#2f9e44",     # functions that call it
    "external": "#868e96",   # stdlib / third-party
    "module": "#7048e8",     # imported modules
    "file": "#495057",       # source file
}

# ===================================================================== #
# Design system
# ===================================================================== #
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#0B0F19; --ink:#030712;
  --glass:rgba(18,24,38,.75); --glass-2:rgba(18,24,38,.5);
  --bd:rgba(255,255,255,.08); --bd-2:rgba(255,255,255,.14);
  --tx:#CBD5E1; --tx-dim:#94A3B8; --tx-faint:#64748B; --tx-hi:#FFFFFF;
  --ac:#0EA5E9; --ac-2:#38BDF8; --teal:#14B8A6;
  --shadow:0 8px 32px rgba(0,0,0,.37);
  --blur:blur(12px);
  --mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:'Geist','Inter',-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
}

.stApp{
  color:var(--tx);font-family:var(--sans);-webkit-font-smoothing:antialiased;
  background-color:#0B0F19;
  background-image:
    radial-gradient(1120px 640px at 12% -8%, rgba(14,165,233,.11), transparent 60%),
    radial-gradient(920px 720px at 108% 2%, rgba(20,184,166,.07), transparent 55%),
    linear-gradient(rgba(255,255,255,.022) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.022) 1px, transparent 1px);
  background-size:auto,auto,46px 46px,46px 46px;
  background-attachment:fixed;
}
[data-testid="stAppViewContainer"],[data-testid="stMain"],.main,
[data-testid="stMainBlockContainer"]{background:transparent!important;}
/* keep the header + toolbar present (the sidebar-expand button lives inside the
   toolbar) but hide only the bits we don't want */
[data-testid="stHeader"]{background:transparent!important;pointer-events:none;}
[data-testid="stHeader"] *{pointer-events:auto;}
[data-testid="stToolbar"]{background:transparent!important;justify-content:flex-start!important;}
[data-testid="stToolbarActions"],[data-testid="stMainMenu"],[data-testid="stMainMenuButton"],
[data-testid="stAppDeployButton"],#MainMenu,footer,[data-testid="stDecoration"],
[data-testid="stStatusWidget"]{display:none!important;}
[data-testid="stMainBlockContainer"],.block-container{
  max-width:1160px;padding:3rem 3rem 8rem!important;}

/* ---- sidebar collapse / expand toggle: ALWAYS visible, top-left ---- */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
button[data-testid="stExpandSidebarButton"],
button[aria-label*="sidebar" i]{
  display:inline-flex!important;visibility:visible!important;opacity:1!important;
  width:34px!important;height:34px!important;min-width:34px!important;align-items:center!important;
  justify-content:center!important;z-index:999999!important;color:#38BDF8!important;
  background:rgba(18,24,38,.8)!important;backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  border:1px solid rgba(56,189,248,.3)!important;border-radius:6px!important;
  box-shadow:0 4px 16px rgba(0,0,0,.35)!important;transition:all .14s ease!important;}
/* the collapsed-state expand button floats at the very top-left of the canvas */
button[data-testid="stExpandSidebarButton"]{position:fixed!important;top:14px!important;left:14px!important;}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"] button:hover,
button[aria-label*="sidebar" i]:hover{
  background:rgba(14,165,233,.2)!important;border-color:var(--ac)!important;
  box-shadow:0 0 18px rgba(14,165,233,.4)!important;}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapseButton"] button svg{color:#38BDF8!important;fill:#38BDF8!important;
  width:18px!important;height:18px!important;}
*{scrollbar-width:thin;scrollbar-color:var(--bd-2) transparent;}
::-webkit-scrollbar{width:9px;height:9px;}
::-webkit-scrollbar-thumb{background:var(--bd-2);border-radius:9px;}

h1,h2,h3,h4{font-family:var(--sans);letter-spacing:-.02em;color:var(--tx-hi);font-weight:600;}
a{color:var(--ac-2);text-decoration:none;}
hr.rule{height:1px;background:var(--bd);margin:28px 0;border:0;}
hr.turnsep{height:1px;background:linear-gradient(90deg,transparent,var(--bd-2),transparent);
  margin:46px 0 38px;border:0;}

/* ---- glass mixin utility ---- */
.glass{background:var(--glass);backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);
  border:1px solid var(--bd);box-shadow:var(--shadow);}

/* ---- topbar ---- */
.topbar{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;
  background:var(--glass);backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);
  border:1px solid var(--bd);border-radius:16px;box-shadow:var(--shadow);
  padding:16px 22px;margin-bottom:34px;}
.brand{font-size:15px;font-weight:600;letter-spacing:-.01em;color:var(--tx-hi);
  display:flex;align-items:center;gap:9px;}
.brand .dot{color:var(--ac);font-size:12px;filter:drop-shadow(0 0 6px rgba(14,165,233,.6));}
.sub{color:var(--tx-faint);font-size:12px;margin-top:6px;}
.pills{display:flex;gap:8px;flex-wrap:wrap;}
.topbar .pills{justify-content:flex-end;}
.pill{font-family:var(--mono);font-size:10px;font-weight:500;letter-spacing:.06em;
  padding:5px 10px;border-radius:7px;border:1px solid var(--bd);background:rgba(3,7,18,.55);
  color:var(--tx-dim);display:inline-flex;align-items:center;gap:7px;white-space:nowrap;
  text-transform:uppercase;}
.pill .led{width:6px;height:6px;border-radius:50%;background:var(--tx-faint);flex:none;}
.pill.ok{color:#7FE7D9;border-color:rgba(20,184,166,.35);}
.pill.ok .led{background:var(--teal);box-shadow:0 0 8px rgba(20,184,166,.9);}
.pill.warn{color:#F0D6A8;border-color:rgba(224,164,74,.32);}
.pill.warn .led{background:#E0A44A;box-shadow:0 0 7px rgba(224,164,74,.7);}
.pill.accent{color:var(--ac-2);border-color:rgba(56,189,248,.32);background:rgba(14,165,233,.12);
  box-shadow:0 0 14px rgba(14,165,233,.12);}

/* ---- eyebrow ---- */
.eyebrow{font-family:var(--mono);font-size:10px;font-weight:500;letter-spacing:.2em;
  text-transform:uppercase;color:var(--tx-faint);margin:0 0 12px;}

/* ---- question + answer ---- */
.qline{font-size:16.5px;font-weight:600;color:var(--tx-hi);line-height:1.55;letter-spacing:-.011em;
  background:rgba(255,255,255,.025);border:1px solid var(--bd);border-left:2px solid var(--ac);
  border-radius:10px;padding:12px 16px;margin:2px 0 26px;}
.answer{font-size:15.5px;line-height:1.82;color:var(--tx-dim);max-width:760px;
  background:var(--glass);backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);
  border:1px solid var(--bd);border-radius:16px;box-shadow:var(--shadow);padding:22px 26px;}
.answer p{margin:0 0 15px;}
.answer p:last-child{margin-bottom:0;}
.answer ul,.answer ol{margin:2px 0 15px;padding-left:22px;}
.answer li{margin:6px 0;}
.answer strong{color:var(--tx-hi);font-weight:600;}
.answer code{font-family:var(--mono);font-size:.82em;background:rgba(14,165,233,.1);
  border:1px solid rgba(56,189,248,.22);border-radius:5px;padding:1.5px 5px;color:#7DD3FC;}
.cite{background:rgba(14,165,233,.15);color:#38BDF8;border:1px solid rgba(56,189,248,.3);
  font-family:var(--mono);padding:2px 6px;border-radius:4px;font-size:.78em;white-space:nowrap;
  box-shadow:0 0 12px rgba(14,165,233,.15);}
.meta{font-family:var(--mono);font-size:10px;color:var(--tx-faint);letter-spacing:.07em;
  margin-top:16px;text-transform:uppercase;}

/* ---- hero (empty state) ---- */
.hero{max-width:560px;margin:10vh auto 0;text-align:center;}
.hero-mark{font-size:26px;color:var(--ac);margin-bottom:24px;
  filter:drop-shadow(0 0 14px rgba(14,165,233,.55));}
.hero-title{font-size:31px;font-weight:600;letter-spacing:-.03em;margin:0 0 14px;color:var(--tx-hi);}
.hero-sub{font-size:14.5px;line-height:1.75;color:var(--tx-dim);margin:0 auto 26px;max-width:450px;}
.hero .pills{justify-content:center;margin-bottom:24px;}
.hero-hint{font-family:var(--mono);font-size:11px;color:var(--tx-faint);letter-spacing:.06em;}

/* ---- legend ---- */
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:0 0 12px;}
.legend .lg{font-family:var(--mono);font-size:10px;color:var(--tx-dim);text-transform:uppercase;
  letter-spacing:.06em;display:flex;align-items:center;gap:6px;}
.legend .lg i{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none;}

/* ---- sidebar (workflow-node modules) ---- */
[data-testid="stSidebar"]{background:var(--glass)!important;backdrop-filter:var(--blur);
  -webkit-backdrop-filter:var(--blur);border-right:1px solid var(--bd);}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding:1.7rem 1.25rem 2rem;}
.side-brand{font-size:14px;font-weight:600;color:var(--tx-hi);display:flex;align-items:center;gap:8px;
  letter-spacing:-.01em;}
.side-brand .dot{color:var(--ac);font-size:11px;filter:drop-shadow(0 0 5px rgba(14,165,233,.6));}
.side-sub{color:var(--tx-faint);font-size:10.5px;margin:6px 0 24px;font-family:var(--mono);letter-spacing:.06em;}
[data-testid="stSidebar"] .eyebrow{margin-top:20px;}
[data-testid="stSidebar"] hr{border-color:var(--bd)!important;margin:18px 0!important;}

/* ---- inputs / selects (dark ink field, glowing focus) ---- */
[data-baseweb="input"],[data-baseweb="select"]>div,[data-baseweb="base-input"]{
  background:var(--ink)!important;border:1px solid var(--bd)!important;border-radius:10px!important;
  transition:border-color .14s ease,box-shadow .14s ease;}
[data-baseweb="input"]:hover,[data-baseweb="select"]>div:hover{border-color:rgba(56,189,248,.4)!important;}
[data-baseweb="input"]:focus-within,[data-baseweb="select"]>div:focus-within{
  border:1px solid var(--ac)!important;
  box-shadow:0 0 0 3px rgba(14,165,233,.16),0 0 18px rgba(14,165,233,.2)!important;}
input,textarea{color:var(--tx)!important;background:transparent!important;font-family:var(--sans)!important;}
input::placeholder,textarea::placeholder{color:var(--tx-faint)!important;}
[data-baseweb="select"] *{color:var(--tx)!important;}
[data-baseweb="popover"] [role="listbox"],[data-baseweb="menu"]{
  background:rgba(3,7,18,.96)!important;backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);
  border:1px solid var(--bd-2)!important;border-radius:12px!important;box-shadow:var(--shadow)!important;}
[data-baseweb="menu"] li:hover{background:rgba(14,165,233,.14)!important;color:var(--ac-2)!important;}
[data-testid="stWidgetLabel"] p,label p{color:var(--tx-dim)!important;font-size:11.5px!important;
  font-weight:500!important;letter-spacing:.01em;}

/* ---- buttons (workflow nodes w/ left marker + hover glow) ---- */
.stButton>button,[data-testid="stBaseButton-secondary"]{
  border-radius:10px!important;border:1px solid var(--bd-2)!important;background:rgba(3,7,18,.55)!important;
  color:var(--tx)!important;font-family:var(--sans)!important;font-weight:500!important;
  font-size:12.5px!important;padding:.55rem .95rem!important;transition:all .14s ease!important;
  box-shadow:none!important;display:flex!important;align-items:center!important;
  justify-content:flex-start!important;gap:9px!important;}
.stButton>button::before,[data-testid="stBaseButton-secondary"]::before{
  content:"";width:6px;height:6px;border-radius:2px;background:var(--tx-faint);
  flex:none;transition:all .14s ease;}
.stButton>button:hover,[data-testid="stBaseButton-secondary"]:hover{
  border-color:var(--ac)!important;background:rgba(14,165,233,.1)!important;color:var(--tx-hi)!important;
  box-shadow:0 0 0 1px rgba(14,165,233,.35),0 0 22px rgba(14,165,233,.22)!important;}
.stButton>button:hover::before{background:var(--ac);box-shadow:0 0 10px var(--ac);}
[data-testid="stBaseButton-primary"]{
  background:linear-gradient(180deg,#0EA5E9,#0284C7)!important;
  border:1px solid rgba(56,189,248,.5)!important;color:#fff!important;font-weight:600!important;
  box-shadow:0 0 24px rgba(14,165,233,.3)!important;}
[data-testid="stBaseButton-primary"]::before{background:#fff!important;
  box-shadow:0 0 10px rgba(255,255,255,.85)!important;}
[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.08)!important;
  box-shadow:0 0 34px rgba(14,165,233,.45)!important;}

/* ---- expanders (glass citation cards + modal trigger) ---- */
[data-testid="stExpander"]{border:1px solid var(--bd)!important;border-radius:12px!important;
  background:var(--glass-2)!important;backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);
  margin-bottom:8px!important;overflow:hidden;box-shadow:0 4px 18px rgba(0,0,0,.25)!important;}
[data-testid="stExpander"] details,[data-testid="stExpander"] summary{background:transparent!important;}
[data-testid="stExpander"] summary{padding:12px 15px!important;font-size:12px!important;
  color:var(--tx-dim)!important;font-family:var(--mono)!important;transition:background .12s,color .12s;}
[data-testid="stExpander"] summary:hover{background:rgba(14,165,233,.06)!important;color:var(--tx)!important;}
[data-testid="stExpander"] summary p{font-family:var(--mono)!important;font-size:12px!important;}
[data-testid="stExpander"] summary code{background:rgba(14,165,233,.14);border:1px solid rgba(56,189,248,.3);
  border-radius:4px;padding:1px 5px;color:var(--ac-2);}
[data-testid="stExpanderDetails"]{padding:6px 15px 15px!important;}

/* ---- code blocks ---- */
[data-testid="stCode"],pre,.stCodeBlock,[data-testid="stCodeBlock"]{
  background:var(--ink)!important;border:1px solid var(--bd)!important;border-radius:10px!important;}
pre code,code[class*="language-"]{font-family:var(--mono)!important;font-size:12px!important;}

/* ---- bordered container (glass graph card) ---- */
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--bd)!important;
  border-radius:16px!important;background:var(--glass)!important;backdrop-filter:var(--blur);
  -webkit-backdrop-filter:var(--blur);box-shadow:var(--shadow)!important;}
[data-testid="stVerticalBlockBorderWrapper"]>div>[data-testid="stVerticalBlock"]{
  padding:15px!important;gap:.6rem!important;}

/* ---- chat input (glass dock) ---- */
[data-testid="stBottomBlockContainer"]{background:linear-gradient(to top,#0B0F19 62%,transparent)!important;
  padding-bottom:1.6rem!important;}
[data-testid="stChatInput"]{background:var(--glass)!important;backdrop-filter:var(--blur);
  -webkit-backdrop-filter:var(--blur);border:1px solid var(--bd)!important;border-radius:14px!important;
  box-shadow:0 12px 40px rgba(0,0,0,.5)!important;max-width:1096px;}
[data-testid="stChatInput"]:focus-within{border-color:var(--ac)!important;
  box-shadow:0 0 0 3px rgba(14,165,233,.16),0 12px 40px rgba(0,0,0,.5)!important;}
[data-testid="stChatInput"] textarea{color:var(--tx)!important;font-family:var(--sans)!important;}
[data-testid="stChatInput"] textarea::placeholder{color:var(--tx-faint)!important;}

/* ---- dialog / modal (heavy glass) ---- */
[data-testid="stDialog"] div[role="dialog"]{background:rgba(11,15,25,.85)!important;
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border:1px solid var(--bd-2)!important;border-radius:18px!important;
  box-shadow:0 30px 100px rgba(0,0,0,.7)!important;}
[data-testid="stDialog"] h1,[data-testid="stDialog"] h2{font-size:13px!important;
  font-family:var(--mono)!important;letter-spacing:.14em!important;text-transform:uppercase;
  color:var(--tx-dim)!important;font-weight:500!important;}

/* ---- misc ---- */
[data-testid="stAlert"]{border-radius:12px!important;border:1px solid var(--bd)!important;
  background:rgba(3,7,18,.6)!important;backdrop-filter:var(--blur);color:var(--tx-dim)!important;}
[data-testid="stSpinner"] p{color:var(--ac-2)!important;font-family:var(--mono)!important;font-size:12px!important;}
[data-testid="stCaptionContainer"] p{color:var(--tx-faint)!important;}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ===================================================================== #
# Helpers
# ===================================================================== #
def indexed_repos() -> list[str]:
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(GRAPH_DIR, "*.sqlite")))


def graph_stats(repo_id: str) -> dict | None:
    path = os.path.join(GRAPH_DIR, f"{repo_id}.sqlite")
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    try:
        c = conn.execute
        return {
            "files": c("SELECT COUNT(*) FROM files").fetchone()[0],
            "symbols": c("SELECT COUNT(*) FROM nodes WHERE symbol_type!='file'").fetchone()[0],
            "calls": c("SELECT COUNT(*) FROM edges WHERE edge_type='CALLS'").fetchone()[0],
        }
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def pill(label: str, kind: str = "", led: bool = False) -> str:
    dot = '<span class="led"></span>' if led else ""
    return f'<span class="pill {kind}">{dot}{html.escape(label)}</span>'


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


_CITE_RE = re.compile(r"\[([\w./\\-]+:\d+(?:[-–]\d+)?)\]")


def accent_citations(md: str) -> str:
    """Turn bare `[path:start-end]` markers in the answer into accent chips."""
    return _CITE_RE.sub(lambda m: f'<span class="cite">{m.group(1)}</span>', md)


def legend_html() -> str:
    rows = [("seed", "#e8590c"), ("callee", "#1c7ed6"), ("caller", "#2f9e44"),
            ("external", "#868e96"), ("module", "#7048e8")]
    dots = "".join(f'<span class="lg"><i style="background:{c}"></i>{n}</span>' for n, c in rows)
    return f'<div class="legend">{dots}</div>'


# ===================================================================== #
# PyVis rendering (physics runs only to stabilise, then freezes)
# ===================================================================== #
_PYVIS_OPTIONS = {
    # Map / photo-gallery style navigation: the graph only captures the wheel and
    # keyboard once clicked, so scrolling the page never fights with zooming.
    "clickToUse": True,
    "physics": {
        "enabled": True,
        "stabilization": {"enabled": True, "iterations": 100, "fit": True},
        "barnesHut": {"gravitationalConstant": -4200, "springLength": 130,
                      "springConstant": 0.03, "damping": 0.6, "avoidOverlap": 0.4},
        "minVelocity": 0.75,
    },
    "interaction": {
        "hover": True,
        "dragNodes": True,
        "dragView": True,
        "zoomView": True,
        "keyboard": {"enabled": True, "bindToWindow": False},
        "navigationButtons": True,
        "tooltipDelay": 120,
    },
    "nodes": {
        "shape": "dot",
        "borderWidth": 2,
        "font": {"size": 14, "color": "#f8f9fa", "face": "Inter,system-ui,sans-serif",
                 "strokeWidth": 4, "strokeColor": "#0b0f19"},
    },
    "edges": {
        "color": {"inherit": False},
        "smooth": {"enabled": True, "type": "dynamic"},
        "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}},
        "width": 1.2,
    },
}

_FREEZE_JS = """
<style>
  body { margin: 0; padding: 0; overflow: hidden; background: #0b0f19; }
  #mynetwork, .card { border: none !important; box-shadow: none !important; }
  h1:empty, h2:empty { display: none !important; }
</style>
<script type="text/javascript">
(function () {
  function settle() {
    try { network.setOptions({ physics: false }); network.fit({ animation: false }); } catch (e) {}
  }
  function freeze() {
    if (typeof network === "undefined" || network === null) { return setTimeout(freeze, 60); }
    network.once("stabilizationIterationsDone", settle);
    setTimeout(settle, 2500);
    window.addEventListener("resize", function () { try { network.fit({ animation: false }); } catch (e) {} });
    network.on("doubleClick", function (params) {
      if (params.nodes && params.nodes.length) {
        network.focus(params.nodes[0], { scale: 1.3, animation: { duration: 400, easingFunction: "easeInOutQuad" } });
      } else {
        network.fit({ animation: { duration: 400, easingFunction: "easeInOutQuad" } });
      }
    });
  }
  freeze();
})();
</script>
""".strip()


def _short(text: str, limit: int = 16) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _pyvis_html(graph: dict, height_px: int) -> str:
    from pyvis.network import Network

    net = Network(height=f"{height_px}px", width="100%", directed=True,
                  bgcolor="#0b0f19", font_color="#f8f9fa",
                  cdn_resources="in_line", notebook=False)
    net.set_options(json.dumps(_PYVIS_OPTIONS))

    for n in graph["nodes"]:
        loc = f"\n{n['file']}:{n['start_line']}-{n['end_line']}" if n["file"] else ""
        net.add_node(
            n["id"],
            label=_short(n["label"]),
            title=f"{n['label']} · {n['role']}{loc}",
            color=ROLE_COLOR.get(n["role"], "#adb5bd"),
            size=24 if n["role"] == "seed" else 14,
        )
    for e in graph["edges"]:
        is_import = e["label"] == "IMPORTS"
        net.add_edge(
            e["source"], e["target"],
            title=e["label"],
            color="#4b4b55" if is_import else "#7a7a86",
            dashes=is_import,
        )
    return net.generate_html(notebook=False).replace("</body>", _FREEZE_JS + "\n</body>")


def render_graph(graph: dict, height: int = 340) -> None:
    if not graph["nodes"]:
        st.caption("No call-graph neighbours for this result.")
        return
    components.html(_pyvis_html(graph, height), height=height, scrolling=False)


@st.dialog("Call graph", width="large")
def graph_dialog(graph: dict) -> None:
    st.markdown(legend_html(), unsafe_allow_html=True)
    components.html(_pyvis_html(graph, 600), height=600, scrolling=False)


# ===================================================================== #
# State
# ===================================================================== #
for key, default in [("repo_id", None), ("repo_url", None), ("history", [])]:
    st.session_state.setdefault(key, default)


# ===================================================================== #
# Sidebar
# ===================================================================== #
with st.sidebar:
    st.markdown('<div class="side-brand"><span class="dot">◆</span> Codebase QA</div>'
                '<div class="side-sub">local call-graph rag</div>', unsafe_allow_html=True)

    eyebrow("Workspace")
    existing = indexed_repos()
    options = ["Select an index…", *existing]
    cur = st.session_state.repo_id
    sel = st.selectbox("index", options,
                       index=options.index(cur) if cur in options else 0,
                       label_visibility="collapsed")
    if sel != "Select an index…" and st.button("Open index", use_container_width=True,
                                                    type="primary"):
        st.session_state.repo_id = sel
        st.session_state.repo_url = sel
        st.session_state.history = []
        st.rerun()

    with st.expander("Add a new index"):
        url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo",
                            label_visibility="collapsed")
        if st.button("Clone & index", use_container_width=True) and url:
            try:
                with st.spinner("cloning → parsing AST → building graph → caching embeddings"):
                    rid = index_repo(url)
                st.session_state.repo_id = rid
                st.session_state.repo_url = url
                st.session_state.history = []
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Indexing failed: {exc}")

    eyebrow("Model")
    models = ollama_models()
    if models:
        pref = next((m for m in models if m.split(":")[0] in ("qwen2.5-coder", "llama3.2", "llama3.1")),
                    models[0])
        model = st.selectbox("model", models, index=models.index(pref),
                             label_visibility="collapsed")
    else:
        model = None
        st.warning("Ollama unreachable on :11434. Retrieval + graph still work.")

    if st.session_state.repo_id:
        st.divider()
        eyebrow("Session")
        st.caption(f"`{st.session_state.repo_url}`")
        if st.button("Clear session", use_container_width=True):
            for k in ("repo_id", "repo_url", "history"):
                st.session_state[k] = None if k != "history" else []
            st.rerun()


# ===================================================================== #
# Topbar
# ===================================================================== #
def render_topbar() -> None:
    stats = graph_stats(st.session_state.repo_id) if st.session_state.repo_id else None
    tags = [pill("System online", "ok", led=True)]
    tags.append(pill(f"Ollama · {len(models)} models", "ok", led=True) if models
                else pill("Ollama offline", "warn", led=True))
    if model:
        tags.append(pill(f"Model {model.split(':')[0]}", "accent"))
    if stats:
        tags.append(pill(f"Index {st.session_state.repo_id} · {stats['symbols']} sym "
                         f"· {stats['calls']} calls", "accent"))
    else:
        tags.append(pill("No index", ""))
    st.markdown(
        '<div class="topbar">'
        '<div><div class="brand"><span class="dot">◆</span> Codebase QA</div>'
        '<div class="sub">AST-aligned retrieval · hybrid rerank · call-graph context · 100% local</div></div>'
        f'<div class="pills">{"".join(tags)}</div>'
        '</div>', unsafe_allow_html=True)


render_topbar()


# ===================================================================== #
# Empty state
# ===================================================================== #
if not st.session_state.repo_id:
    st.markdown(
        '<div class="hero">'
        '<div class="hero-mark">◆</div>'
        '<div class="hero-title">Ask your codebase.</div>'
        '<div class="hero-sub">Point the engine at a repository and query it in natural '
        'language. Every answer is grounded in real AST spans and a live call-graph.</div>'
        '<div class="pills">'
        + (pill("Ollama ready", "ok", True) if models else pill("Ollama offline", "warn", True))
        + pill(f"{len(models)} local models" if models else "No models", "")
        + pill(f"{len(existing)} indexes available", "")
        + '</div>'
        '<div class="hero-hint">select or add an index in the left panel to begin →</div>'
        '</div>', unsafe_allow_html=True)
    st.stop()


# ===================================================================== #
# Query
# ===================================================================== #
question = st.chat_input("Ask about this codebase…")
if question:
    try:
        with st.spinner("retrieving → re-ranking → generating locally"):
            data = answer_question(st.session_state.repo_id, question, model=model)
    except RuntimeError as exc:
        st.warning(f"{exc}\n\nShowing retrieved context + call-graph without a generated answer.")
        data = answer_question(st.session_state.repo_id, question, model=model, generate=False)
    st.session_state.history.append(data)


# ===================================================================== #
# Conversation
# ===================================================================== #
for i, turn in enumerate(reversed(st.session_state.history)):
    if i:
        st.markdown('<hr class="turnsep">', unsafe_allow_html=True)

    st.markdown(f'<div class="qline">{html.escape(turn["question"])}</div>',
                unsafe_allow_html=True)

    if turn.get("answer"):
        eyebrow("Response")
        st.markdown(f'<div class="answer">\n\n{accent_citations(turn["answer"])}\n\n</div>',
                    unsafe_allow_html=True)
        b = turn["backends"]
        st.markdown(
            f'<div class="meta">{html.escape(turn["model"])} &nbsp;·&nbsp; '
            f'embed {html.escape(b["embed"].split(":")[-1])} &nbsp;·&nbsp; '
            f'rerank {html.escape(b["rerank"].split(":")[-1])}</div>', unsafe_allow_html=True)
    else:
        st.caption("No local model was available to generate an answer.")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    cites = turn["citations"]
    eyebrow(f"Sources · {len(cites)}")
    top = max((c["rerank_score"] for c in cites), default=0.0) or 1.0
    for c in cites:
        filled = max(1, round(6 * c["rerank_score"] / top))
        bar = "▰" * filled + "▱" * (6 - filled)
        label = (f"`{c['file']}:{c['start_line']}-{c['end_line']}`  ·  "
                 f"{c['symbol_name']}  ·  {bar}  {c['rerank_score']:.3f}")
        with st.expander(label):
            st.code(c["code"], language=c["language"] or "text", line_numbers=True)

    g = turn["graph"]
    st.markdown('<hr class="rule">', unsafe_allow_html=True)
    eyebrow("Call graph")
    with st.container(border=True):
        st.markdown(legend_html(), unsafe_allow_html=True)
        render_graph(g, height=330)
        if g["nodes"] and st.button("Open interactive graph  ↗",
                                    key=f"gopen{i}", use_container_width=True):
            graph_dialog(g)
