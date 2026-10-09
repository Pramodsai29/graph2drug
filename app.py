"""
Graph2Drug — web demo.

    streamlit run app.py

Runs the trained D-GCAN checkpoints on CPU (no GPU needed). Locally, or deployed on
Streamlit Community Cloud from the GitHub repo (see DEPLOY.md).
"""
import base64
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
import atom_explain as ax  # noqa: E402
import demo_predict as dp  # noqa: E402

RESULTS = ROOT / "results"
DIAGRAMS = ROOT / "docs" / "diagrams"
GITHUB = "https://github.com/Pramodsai29/graph2drug"
INDIGO, SKY, CORAL, GRAY = "#4285f4", "#9b72cb", "#d96570", "#9aa0a6"  # Google / Gemini palette

TASKS = {
    "BBBP": {
        "short": "Brain penetration",
        "title": "Blood–brain barrier penetration",
        "question": "Can this molecule cross the blood–brain barrier and reach the brain?",
        "examples": {
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Diazepam (Valium)": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
            "Imipramine (antidepressant)": "CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31",
            "Penicillin G (antibiotic)": "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O",
            "Sucrose (table sugar)": "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O",
        },
    },
    "BACE": {
        "short": "Alzheimer's target",
        "title": "BACE-1 inhibition (Alzheimer's disease)",
        "question": "Does this molecule inhibit BACE-1, an enzyme linked to Alzheimer's disease?",
        "examples": {
            "Test-set inhibitor A": "Cc1cc(C2(c3cccc(C#CC4CC4)c3)N=C(N)c3c(F)cccc32)cn(C)c1=O",
            "Test-set inhibitor B": "CN(C(=O)CCc1cc2ccccc2nc1N)C1CCCCC1",
            "Test-set non-inhibitor A": "Nc1nc(CCc2ccc3cc[nH]c3c2)cc(=O)[nH]1",
            "Test-set non-inhibitor B": "CC[C@@H](CC(=O)NC1C2CC3CC(C2)CC1C3)n1c(N)nc2cc(Cl)ccc21",
        },
    },
    "druglikeRandom": {
        "short": "Drug-likeness",
        "title": "Drug-likeness (the original D-GCAN paper's task)",
        "question": "Does this molecule look like an approved drug, or like a random screening compound?",
        "examples": {
            "Ibuprofen (painkiller)": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "Metformin (diabetes)": "CN(C)C(=N)N=C(N)N",
            "Test-set ZINC compound A": "O=C(COc1ccccc1/C=C1\\N=C2CCCCCN2C1=O)N1CCCC1",
            "Test-set ZINC compound B": "CCCCc1ccc(NC(=S)N2CCN(c3ccc(OC)cc3)CC2)cc1",
            "Test-set ZINC compound C": "CC(C)Cc1ccc([C@H](C)NC(=O)CSc2nnc(N3CCCC3)s2)cc1",
        },
    },
}

LOGO_SVG = (ROOT / "docs" / "brand" / "graph2drug_logo.svg").read_text()
LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(LOGO_SVG.encode()).decode()

st.set_page_config(page_title="Graph2Drug", page_icon=LOGO_URI, layout="wide",
                   initial_sidebar_state="collapsed")

# light / dark = Streamlit's active theme ([theme.light] / [theme.dark] in config.toml), so our
# colours and Streamlit's own widgets always match
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
PAL = ({"bg": "#131314", "ink": "#e3e3e3", "muted": "#c4c7c5", "surface": "#1e1f20", "surface2": "#282a2c",
        "outline": "#444746", "blue": "#a8c7fa", "blue_soft": "#004a77", "blue_ink": "#c2e7ff",
        "ok_bg": "#0f5223", "ok_ink": "#c4eed0", "bad_bg": "#8c1d18", "bad_ink": "#f9dedc",
        "pos": "#6dd58c", "neg": "#f2b8b5", "gauge": "linear-gradient(90deg, #8c1d18 0%, #4a4458 45%, #0f5223 100%)"}
       if DARK else
       {"bg": "#ffffff", "ink": "#1f1f1f", "muted": "#444746", "surface": "#f0f4f9", "surface2": "#e9eef6",
        "outline": "#c4c7c5", "blue": "#0b57d0", "blue_soft": "#d3e3fd", "blue_ink": "#041e49",
        "ok_bg": "#c4eed0", "ok_ink": "#0d5223", "bad_bg": "#f9dedc", "bad_ink": "#8c1d18",
        "pos": "#146c2e", "neg": "#b3261e", "gauge": "linear-gradient(90deg, #f9dedc 0%, #e8def8 45%, #c4eed0 100%)"})
MOL_BG = tuple(int(PAL["surface"][i:i + 2], 16) / 255 for i in (1, 3, 5))  # molecule pictures sit on the card colour

# ------------------------------------------------------------------ style (Gemini-like)
st.html("<style>:root {" + "".join(f"--{k.replace('_', '-')}:{v};" for k, v in PAL.items()) + "}</style>")
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Google+Sans+Code&display=swap');
html, body, .stApp { background: var(--bg); color: var(--ink);
        font-family: 'Google Sans', 'Inter', system-ui, -apple-system, sans-serif; }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp textarea, .stMarkdown,
h1, h2, h3, h4, [data-testid="stCaptionContainer"] { font-family: 'Google Sans', 'Inter', system-ui, sans-serif !important; }
code, pre { font-family: 'Google Sans Code', ui-monospace, monospace !important; }
h1, h2, h3, h4 { font-weight: 500 !important; letter-spacing: -0.01em; color: var(--ink); }
.block-container { padding-top: 1.4rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
/* Streamlit's (invisible) top toolbar spans the page width: let clicks through to our header,
   except on its own menu button */
header[data-testid="stHeader"], [data-testid="stToolbar"], .stAppToolbar { pointer-events: none; }
[data-testid="stToolbar"] button, .stAppToolbar button, [data-testid="stMainMenu"] { pointer-events: auto; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

/* top bar */
.topbar { display:flex; align-items:center; justify-content: space-between; padding: 4px 4px 14px; }
.brand { display:flex; align-items:center; gap:12px; font-size: 32px; font-weight: 500; color: var(--ink); letter-spacing: -0.01em; }
.brand img { width: 46px; height: 46px; }
.topbar a.gh { text-decoration:none; color: var(--blue); background: var(--surface); border-radius: 999px; padding: 10px 20px; font-size: 14px; font-weight: 500; }
.topbar a.gh:hover { background: var(--blue-soft); }
.topbar .actions { display:flex; align-items:center; gap: 10px; }
.topbar .mode { width: 42px; height: 42px; border-radius: 12px; border: 1px solid var(--outline); background: var(--surface);
        color: var(--ink); font-size: 18px; line-height: 1; cursor: pointer; padding: 0; display:flex; align-items:center;
        justify-content:center; transition: background .2s; }
.topbar .mode:hover { background: var(--surface2); }

/* headline with Gemini's moving gradient */
.greet h1 { font-size: 52px; line-height: 1.15; margin: 22px 0 0; font-weight: 500 !important; max-width: 1000px;
        background: linear-gradient(74deg, #4285f4 0%, #9b72cb 15%, #d96570 30%, #d96570 36%, #9b72cb 52%, #4285f4 68%, #9b72cb 84%, #4285f4 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent; background-size: 250% auto;
        animation: shimmer 9s linear infinite; }
@keyframes shimmer { to { background-position: 250% center; } }
.lead { color: var(--muted); font-size: 17px; max-width: 820px; margin: 16px 0 24px; line-height: 1.6; }

/* suggestion-style cards */
.sugg { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin-bottom: 12px; }
.sugg div { background: var(--surface); border-radius: 16px; padding: 16px 16px 46px; position: relative; transition: background .2s; }
.sugg div:hover { background: var(--surface2); }
.sugg b { display:block; font-size: 22px; font-weight: 500; color: var(--ink); margin-bottom: 6px; }
.sugg span { color: var(--muted); font-size: 14px; line-height: 1.45; }
.sugg i { position:absolute; right: 12px; bottom: 12px; width: 34px; height: 34px; border-radius: 50%; background: var(--bg);
        display:flex; align-items:center; justify-content:center; font-style: normal; font-size: 17px; }

/* equal-height card rows */
.grid { display:grid; gap: 14px; align-items: stretch; }
.grid.c4 { grid-template-columns: repeat(4, minmax(0,1fr)); }
.grid.c3 { grid-template-columns: repeat(3, minmax(0,1fr)); }
.grid.c32 { grid-template-columns: 3fr 2fr; }
@media (max-width: 900px) { .sugg, .grid.c4, .grid.c3, .grid.c32 { grid-template-columns: repeat(1, minmax(0,1fr)); }
        .greet h1 { font-size: 36px; } .brand { font-size: 26px; } }

/* tabs = Material 3 navigation pills */
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; border: none; padding: 4px 0; }
.stTabs [role="tab"] { height: 40px; padding: 0 18px; border-radius: 999px; color: var(--muted); background: transparent; border-bottom: none !important; }
.stTabs [role="tab"]:hover { background: var(--surface); }
.stTabs [role="tab"][aria-selected="true"] { background: var(--blue-soft); }
.stTabs [role="tab"] p { font-size: 14.5px; font-weight: 500; }
.stTabs [role="tab"][aria-selected="true"] p { color: var(--blue-ink); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [role="tablist"] { border-bottom: none !important; box-shadow: none !important; }
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before, .stTabs [role="tab"]::after { display: none !important; }
.stTabs .react-aria-SelectionIndicator { display: none !important; }

/* option pills (segmented controls): separate chips with a gap */
[data-testid="stButtonGroup"] [role="radiogroup"] { gap: 10px !important; flex-wrap: wrap; }
[data-testid="stButtonGroup"] button { border-radius: 999px !important; margin: 0 !important; padding: 6px 18px !important;
        border: 1px solid var(--outline) !important; background: transparent !important; color: var(--muted) !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--blue-soft) !important; border-color: transparent !important;
        color: var(--blue-ink) !important; }
[data-testid="stButtonGroup"] button p { font-weight: 500; }

/* surfaces */
.g-card { background: var(--surface); border-radius: 24px; padding: 20px 22px; height: 100%; box-sizing: border-box; }
.g-card h4 { margin: 0 0 6px; font-size: 18px; color: var(--ink); }
.g-card p { color: var(--muted); margin: 0; font-size: 14.5px; line-height: 1.55; }
.g-card b { color: var(--ink); }
.g-step { font-weight: 500; font-size: 12.5px; color: var(--blue-ink); background: var(--blue-soft); border-radius: 999px;
        padding: 4px 11px; display:inline-block; margin-bottom: 12px; }
.g-big { font-size: clamp(20px, 1.8vw, 28px); white-space: nowrap; font-weight: 500; margin: 6px 0 6px; background: linear-gradient(74deg, #4285f4, #9b72cb 55%, #d96570);
        -webkit-background-clip: text; background-clip: text; color: transparent; }
.g-section { font-size: 22px; font-weight: 500; margin: 30px 0 12px; color: var(--ink); }
.g-muted { color: var(--muted); font-size: 14px; }
.g-muted b { color: var(--ink); }

/* the "Ask Gemini" style input */
div[data-testid="stTextInput"] input { background: var(--surface) !important; border: none !important; border-radius: 999px !important;
        padding: 14px 22px !important; font-size: 16px !important; color: var(--ink) !important; }
div[data-testid="stTextInput"] > div > div { border: none !important; background: transparent !important; }
div[data-testid="stTextInput"] input:focus { background: var(--surface2) !important; }
div[data-baseweb="select"] > div { background: var(--surface) !important; border: none !important; border-radius: 999px !important; padding-left: 8px; }
[data-testid="stSelectbox"] [role="group"] { background: var(--surface) !important; border: none !important; border-radius: 999px !important; color: var(--ink) !important; }
.stButton button, .stDownloadButton button { border-radius: 999px !important; font-weight: 500 !important; padding: 8px 20px !important; }
.stButton button[kind="primary"] { background: var(--blue) !important; border: none !important; color: var(--bg) !important; }

/* model answer, like a Gemini response */
.answer { display:flex; gap: 14px; align-items:flex-start; margin-bottom: 14px; }
.answer img { flex: 0 0 34px; width: 34px; height: 34px; }
.answer .txt { font-size: 22px; line-height: 1.35; color: var(--ink); }
.answer .txt small { display:block; font-size: 13px; color: var(--muted); margin-bottom: 4px; }
.answer .txt b.pos { color: var(--pos); font-weight: 500; }
.answer .txt b.neg { color: var(--neg); font-weight: 500; }
.gauge { position: relative; height: 10px; border-radius: 999px; margin: 12px 0 6px; background: var(--gauge); }
.gauge .fill { position:absolute; top:-5px; width: 20px; height: 20px; border-radius: 50%; margin-left: -10px;
        background: linear-gradient(135deg, #4285f4, #9b72cb, #d96570); box-shadow: 0 0 0 3px var(--surface), 0 1px 4px rgba(0,0,0,.25); }
.gauge .thr { position:absolute; top:-4px; width: 2px; height: 18px; background: var(--muted); opacity: .5; }
.gauge-labels { display:flex; justify-content: space-between; font-size: 12px; color: var(--muted); }
.prob { font-size: 44px; font-weight: 500; line-height: 1; color: var(--ink); }
.chip { display:inline-block; border-radius: 8px; padding: 5px 11px; font-size: 13px; font-weight: 500; margin: 3px 6px 3px 0; }
.chip.ok { background: var(--ok-bg); color: var(--ok-ink); }
.chip.bad { background: var(--bad-bg); color: var(--bad-ink); }
.chip.info { background: var(--blue-soft); color: var(--blue-ink); }
.claim { border-radius: 24px; padding: 18px 20px; }
.claim.yes { background: var(--ok-bg); color: var(--ok-ink); }
.claim.no { background: var(--bad-bg); color: var(--bad-ink); }
.claim b { font-size: 17px; font-weight: 500; }
.claim span { opacity: .85; font-size: 14px; }
.footer { text-align:center; color: var(--muted); font-size: 13px; margin: 44px 0 10px; }
.footer a { color: var(--blue); }
div[data-testid="stMetric"] { background: var(--surface); border-radius: 20px; padding: 14px 18px; }
div[data-testid="stImage"] img { border-radius: 20px; }
div[data-testid="stExpander"] details { border: none; background: var(--surface); border-radius: 20px; }
div[data-testid="stCode"] pre { border-radius: 16px; }
/* no hover "fullscreen" pop-up on pictures */
[data-testid="stElementContainer"]:has([data-testid="stImage"]) [data-testid="stElementToolbar"],
[data-testid="stImage"] ~ button, [data-testid="stFullScreenFrame"] > div > button { display: none !important; }
</style>
""")

LOGO = f'<img src="{LOGO_URI}" alt="">'


def card(html):
    st.html(f'<div class="g-card">{html}</div>')


def cards(items, cls):
    """A row of equal-height cards (one CSS grid instead of Streamlit columns)."""
    st.html(f'<div class="grid {cls}">' + "".join(f'<div class="g-card">{h}</div>' for h in items) + "</div>")


def mol_png(smiles, size=(1040, 720)):
    """High-resolution molecule picture (2x the display size) on the card colour."""
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    if DARK:
        rdMolDraw2D.SetDarkMode(o)
    o.setBackgroundColour(MOL_BG + (1.0,))
    o.bondLineWidth = 3
    o.padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(d, Chem.MolFromSmiles(smiles))
    d.FinishDrawing()
    return d.GetDrawingText()


# ------------------------------------------------------------------ cached model calls
@st.cache_resource(show_spinner="Loading trained model…")
def get_predictor(name):
    return dp.Predictor(name)


@st.cache_data(show_spinner=False, max_entries=256)
def predict_one(task, smiles):
    return get_predictor(task).predict([smiles])[smiles]


@st.cache_data(show_spinner=False, max_entries=256)
def occlusion(task, smiles):
    return ax.occlusion(get_predictor(task), smiles)[1]


@st.cache_data(show_spinner=False, max_entries=128)
def shap_vals(task, smiles):
    return ax.shap_values(get_predictor(task), smiles)[1]


def fmt_pm(m, s):
    return f"{m:.3f} ± {s:.3f}"


def p_fmt(p):
    return "—" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def table(df):
    st.dataframe(df, hide_index=True, width="stretch")


# ------------------------------------------------------------------ top bar + headline
st.html(f"""
<div class="topbar">
  <div class="brand">{LOGO}<span>Graph2Drug</span></div>
  <div class="actions">
    <a class="gh" href="{GITHUB}" target="_blank">View source on GitHub</a>
    <button class="mode" id="g2d-mode" title="{'Switch to light mode' if DARK else 'Switch to dark mode'}"
            data-next="{'Light' if DARK else 'Dark'}">{'☀' if DARK else '☾'}</button>
  </div>
</div>
<div class="greet"><h1>Can a computer tell which molecules could become medicines?</h1></div>
<div class="lead">Graph2Drug predicts drug properties from molecular graphs, and audits the published
  D-GCAN model: we found a hidden training bug, fixed it, and re-tested the paper's claims on five datasets.</div>
<script>
  // Streamlit keeps the viewer's theme choice in localStorage; flip it and reload so Streamlit's
  // widgets and our colours switch together. One delegated listener survives reruns.
  if (!window.__g2dMode) {{
    window.__g2dMode = true;
    document.addEventListener("click", (e) => {{
      const b = e.target.closest && e.target.closest("#g2d-mode");
      if (!b) return;
      try {{ localStorage.setItem("stActiveTheme-/-v2", JSON.stringify(b.dataset.next)); }} catch (err) {{}}
      location.reload();
    }});
  }}
</script>
<div class="sugg">
  <div><b>0.466 → 0.640</b><span>BBBP test AUC after our one-line fix</span><i>🐞</i></div>
  <div><b>5 datasets</b><span>BBBP · BACE · ClinTox · Tox21 · drug-likeness</span><i>🧪</i></div>
  <div><b>18 experiments</b><span>multi-seed runs with significance tests</span><i>📈</i></div>
  <div><b>1 of 2</b><span>paper claims confirmed on its own data</span><i>🔎</i></div>
</div>
""", unsafe_allow_javascript=True)

tab_over, tab_pred, tab_res, tab_find, tab_how, tab_check = st.tabs(
    ["🏠 Overview", "🔬 Predict & explain", "📊 Results", "🔎 Findings", "⚙️ How it works", "✅ Verify"])

# ------------------------------------------------------------------ overview
with tab_over:
    st.html('<div class="g-section">The project in four steps</div>')
    steps = [
        ("Step 1", "Found a hidden bug", "The published D-GCAN code applies sigmoid and then softmax. "
         "On BBBP the model stops learning (AUC 0.466 ≈ coin flip). A one-line fix restores it.", "0.466 → 0.640"),
        ("Step 2", "Tested it properly", "Four MoleculeNet datasets, scaffold splits, 5 training seeds each, "
         "mean ± std instead of one lucky run.", "0.64–0.84"),
        ("Step 3", "Stress-tested the design", "Random vs scaffold split, standard GNN baselines and ablations: "
         "simple GNNs match or beat D-GCAN; no single part is essential.", "3 studies"),
        ("Step 4", "Re-tested the paper", "On the authors' own data, their numbers reproduce. Graph convolution "
         "helps (confirmed); attention does not (not confirmed).", "1 of 2 claims"),
    ]
    cards([f'<span class="g-step">{k}</span><h4>{t}</h4><div class="g-big">{big}</div><p>{d}</p>'
           for k, t, d, big in steps], "c4")

    st.html('<div class="g-section">What is new in our work</div>')
    cards(["""<p style="color:var(--ink);font-size:15px;line-height:1.9">
        <span class="chip info">1</span> A <b>hidden, data-dependent bug</b> in published code — 28 papers cite D-GCAN, none re-tested it.<br>
        <span class="chip info">2</span> The <b>first independent multi-dataset, multi-seed re-evaluation</b> of D-GCAN.<br>
        <span class="chip info">3</span> The paper's <b>attention claim fails</b> even on its own data; the convolution claim holds.<br>
        <span class="chip info">4</span> <b>Evaluation flaws</b> in the original: no validation set, one run, "AUC" from 0/1 labels.<br>
        <span class="chip info">5</span> D-GCAN's <b>attention is not a faithful explanation</b>; occlusion and SHAP are.</p>""",
           """<h4>Why it matters</h4><p>Developing a drug takes 10–15 years and billions of dollars, and
        most candidates fail. Models like D-GCAN screen molecules on a computer first — but only if
        they work as reported. Reproducing and auditing them is how the field finds out.</p>
        <p style="margin-top:12px"><span class="chip ok">reproducible</span><span class="chip ok">open source</span>
        <span class="chip ok">every number traceable to a script</span></p>"""], "c32")

    st.html('<div class="g-section">System architecture</div>')
    st.image(str(DIAGRAMS / "1.5_Architecture_Diagram.png"), width="stretch")

# ------------------------------------------------------------------ predict & explain
with tab_pred:
    task = st.segmented_control("Prediction task", list(TASKS), default="BBBP", key="task",
                                format_func=lambda k: TASKS[k]["short"]) or "BBBP"
    cfg = TASKS[task]
    st.html(f'<div class="g-muted" style="margin:-4px 0 10px">{cfg["title"]} — <b>{cfg["question"]}</b></div>')

    c1, c2 = st.columns([1, 2])
    with c1:
        choice = st.selectbox("Example molecule", ["— type your own —"] + list(cfg["examples"]),
                              index=1, key=f"ex_{task}")
    with c2:
        smiles = st.text_input("SMILES (molecule as text — copy one from PubChem)",
                               value=cfg["examples"].get(choice, ""), key=f"smi_{task}_{choice}")

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if smiles and mol is None:
        st.warning("That isn't a valid SMILES string — RDKit could not read it.")
    if mol is not None:
        pred = get_predictor(task)
        p = predict_one(task, smiles)
        positive = p > dp.THRESHOLD
        a, b, c = st.columns([1.25, 1, 1])
        with a:
            st.image(mol_png(smiles), width="stretch")
        with b:
            st.html(f"""
            <div class="answer">{LOGO}<div class="txt"><small>Graph2Drug says</small>
              This molecule most likely <b class="{'pos' if positive else 'neg'}">{'is ' if task == 'druglikeRandom' else ''}{pred.verdict(p)}</b>.</div></div>
            <div class="g-card"><div class="g-muted">Predicted probability</div>
              <div class="prob">{p:.3f}</div>
              <div class="gauge"><div class="thr" style="left:{dp.THRESHOLD * 100}%"></div>
                   <div class="fill" style="left:{p * 100}%"></div></div>
              <div class="gauge-labels"><span>0</span><span>threshold {dp.THRESHOLD}</span><span>1</span></div>
            </div>""")
        with c:
            mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
            hbd, hba = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            rules = [("MW ≤ 500", mw <= 500), ("logP ≤ 5", logp <= 5), ("H-donors ≤ 5", hbd <= 5),
                     ("H-acceptors ≤ 10", hba <= 10)]
            chips = "".join(f'<span class="chip {"ok" if ok else "bad"}">{"✓" if ok else "✗"} {r}</span>' for r, ok in rules)
            card(f"""<h4>{rdMolDescriptors.CalcMolFormula(mol)}</h4>
            <p>Molecular weight <b>{mw:.1f}</b> · logP <b>{logp:.2f}</b><br>
            Polar surface area <b>{tpsa:.1f} Å²</b> · heavy atoms <b>{mol.GetNumHeavyAtoms()}</b></p>
            <p style="margin-top:12px">Lipinski's rule of five (a chemist's rule of thumb, not used by the model):</p>
            <div style="margin-top:6px">{chips}</div>""")

        st.html('<div class="g-section">Why? Which atoms drove this prediction</div>')
        st.html(f'<div class="g-muted" style="margin-top:-6px;margin-bottom:10px"><span class="chip bad">red</span> '
                f'pushes towards <i>{pred.cfg["positive"]}</i> &nbsp; <span class="chip info">blue</span> pushes away. '
                'An atom is "removed" by zeroing its features and re-running the model.</div>')
        e1, e2 = st.columns(2)
        with e1:
            occ = occlusion(task, smiles)
            st.image(ax.draw(smiles, occ, "Occlusion — one atom at a time", size=(1120, 760), dark=DARK, bg=MOL_BG), width="stretch")
            st.caption("Top atoms: " + ", ".join(f"{s}{i} ({v:+.3f})" for s, i, v in ax.top_atoms(smiles, occ)))
        with e2:
            key = f"shap_on_{task}_{smiles}"
            if st.button("✨ Compute SHAP live (≈5 s)", key=f"btn_{key}", type="primary") or st.session_state.get(key):
                st.session_state[key] = True
                with st.spinner("Estimating Shapley values over atoms…"):
                    sv = shap_vals(task, smiles)
                st.image(ax.draw(smiles, sv, "SHAP — atoms in all combinations", size=(1120, 760), dark=DARK, bg=MOL_BG), width="stretch")
                st.caption("Top atoms: " + ", ".join(f"{s}{i} ({v:+.3f})" for s, i, v in ax.top_atoms(smiles, sv)))
            else:
                card("""<h4>SHAP: fair credit for atoms that work together</h4><p>Occlusion removes one atom
                at a time, so it misses atoms that cover for each other (try <b>sucrose</b> on the brain task:
                eight OH groups). SHAP averages each atom's effect over many combinations of the others.</p>""")

    with st.expander(f"Compare all {cfg['short']} example molecules"):
        pr = get_predictor(task)
        probs = pr.predict(list(cfg["examples"].values()))
        table(pd.DataFrame([{"Molecule": n, "Probability": round(probs[s], 3), "Prediction": pr.verdict(probs[s])}
                            for n, s in cfg["examples"].items()]))

# ------------------------------------------------------------------ results
with tab_res:
    def mean_std(path):
        v = pd.read_csv(path)["test_auc"]
        return v.mean(), v.std()

    rows = [("BBBP", *mean_std(RESULTS / "BBBP_v8_multiseed_results.csv")),
            ("BACE", *mean_std(RESULTS / "BACE_v9_multiseed_results.csv")),
            ("ClinTox", *mean_std(RESULTS / "ClinTox_v9_multiseed_results.csv")),
            ("Tox21", *mean_std(RESULTS / "Tox21_v9_multiseed_results.csv")),
            ("Drug-likeness", *mean_std(RESULTS / "druglikeScaffold_v16_full_multiseed_results.csv"))]
    res = pd.DataFrame(rows, columns=["Dataset", "mean", "std"])
    res["lo"], res["hi"] = res["mean"] - res["std"], res["mean"] + res["std"]
    res["label"] = res.apply(lambda r: f"{r['mean']:.3f}", axis=1)

    cols = st.columns(len(rows) + 1)
    cols[0].metric("Original code · BBBP", "0.466", "coin flip", delta_color="off")
    for col, r in zip(cols[1:], rows):
        col.metric(f"Fixed · {r[0]}", f"{r[1]:.3f}", f"± {r[2]:.3f}", delta_color="off")
    st.caption("Test ROC-AUC on scaffold splits (new core structures in the test set), mean ± std over "
               "training seeds (5; drug-likeness 3). 0.5 = guessing, 1.0 = perfect.")

    left, right = st.columns(2)
    with left:
        card("<h4>Test AUC by dataset — fixed model</h4>")
        base = alt.Chart(res).encode(x=alt.X("Dataset:N", sort=None, title=None,
                                             axis=alt.Axis(labelAngle=0, labelFontSize=12)))
        bars = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=46).encode(
            y=alt.Y("mean:Q", scale=alt.Scale(domain=[0, 1]), title="Test ROC-AUC"),
            color=alt.Color("mean:Q", scale=alt.Scale(range=[SKY, INDIGO]), legend=None),
            tooltip=["Dataset", alt.Tooltip("mean:Q", format=".3f", title="AUC"), alt.Tooltip("std:Q", format=".3f")])
        err = base.mark_rule(color=PAL["muted"]).encode(y="lo:Q", y2="hi:Q")
        text = base.mark_text(dy=-12, color=PAL["ink"], fontWeight="bold").encode(y="hi:Q", text="label:N")
        chance = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[5, 5], color=GRAY).encode(y="y:Q")
        st.altair_chart((bars + err + text + chance).properties(height=330), width="stretch")
    with right:
        card("<h4>BBBP validation AUC while training — original vs fixed</h4>")
        v1 = pd.read_csv(RESULTS / "BBBP_training_log.csv")[["epoch", "valid_auc"]].assign(Model="Original code")
        v5 = pd.read_csv(RESULTS / "BBBP_v5_training_log.csv")[["epoch", "valid_auc"]].assign(Model="Fixed")
        line = alt.Chart(pd.concat([v1, v5])).mark_line(strokeWidth=2.5).encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("valid_auc:Q", title="Validation AUC", scale=alt.Scale(domain=[0.2, 1])),
            color=alt.Color("Model:N", scale=alt.Scale(domain=["Fixed", "Original code"], range=[INDIGO, CORAL]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "epoch", alt.Tooltip("valid_auc:Q", format=".3f")])
        st.altair_chart(line.properties(height=330), width="stretch")

    l2, r2 = st.columns([1, 1])
    with l2:
        card("""<h4>What the original model actually output</h4><p>Every BBBP test molecule scored ≈ 1.0:
        the double squashing saturated the output, gradients fell to about 10⁻¹⁰ and learning stopped
        at epoch 5.</p>""")
    with r2:
        st.image(str(RESULTS / "BBBP_v2_score_histogram.png"), width="stretch")

# ------------------------------------------------------------------ findings
with tab_find:
    st.caption("Every number below is read live from the results files written by the experiment scripts "
               "(experiments/train_dataset_v*.py). p < 0.05 = the difference is unlikely to be chance.")
    section = st.segmented_control(
        "Finding", ["① Split matters", "② Simple GNNs", "③ Ablations", "④ Paper re-test", "⑤ Explanations"],
        default="④ Paper re-test", key="finding") or "④ Paper re-test"

    if section.startswith("①"):
        st.subheader("A random split makes the model look much better than it is")
        d = pd.read_csv(RESULTS / "split_comparison_v10.csv")
        long = pd.concat([pd.DataFrame({"Dataset": d.dataset, "Split": "Scaffold (honest)", "AUC": d.scaffold_auc}),
                          pd.DataFrame({"Dataset": d.dataset, "Split": "Random (easy)", "AUC": d.random_auc})])
        ch = alt.Chart(long).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Dataset:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="Split:N",
            y=alt.Y("AUC:Q", scale=alt.Scale(domain=[0, 1]), title="Test AUC"),
            color=alt.Color("Split:N", scale=alt.Scale(range=[CORAL, INDIGO]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Dataset", "Split", alt.Tooltip("AUC:Q", format=".3f")])
        c1, c2 = st.columns([3, 2])
        with c1:
            st.altair_chart(ch.properties(height=320), width="stretch")
        with c2:
            card("""<h4>Why</h4><p>A <b>scaffold split</b> puts molecules with new core structures into the
            test set, like a genuinely new drug. Those test molecules are less similar to the training set
            (Tanimoto 0.36–0.57 vs 0.48–0.79), so it is the honest test. A random split inflates BBBP from
            0.64 to 0.89. ClinTox reverses: its test set has only 10–15 toxic molecules.</p>""")
        table(pd.DataFrame({"Dataset": d.dataset, "Scaffold AUC": [fmt_pm(m, s) for m, s in zip(d.scaffold_auc, d.scaffold_std)],
                            "Random AUC": [fmt_pm(m, s) for m, s in zip(d.random_auc, d.random_std)], "p": d.welch_p.map(p_fmt)}))

    elif section.startswith("②"):
        st.subheader("Untuned standard GNNs beat D-GCAN on BBBP and match it on BACE")
        d = pd.read_csv(RESULTS / "architecture_comparison_v12.csv")
        ch = alt.Chart(d).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)), column=alt.Column("dataset:N", title=None),
            y=alt.Y("test_auc_mean:Q", scale=alt.Scale(domain=[0.5, 0.9], clamp=True), title="Test AUC"),
            color=alt.condition(alt.datum.model == "D-GCAN (fixed)", alt.value(CORAL), alt.value(INDIGO)),
            tooltip=["model", alt.Tooltip("test_auc_mean:Q", format=".3f")]).properties(width=260, height=280)
        st.altair_chart(ch)
        table(pd.DataFrame({"Dataset": d.dataset, "Model": d.model,
                            "Test AUC": [fmt_pm(m, s) for m, s in zip(d.test_auc_mean, d.test_auc_std)],
                            "p vs D-GCAN": d.welch_p_vs_dgcan.map(p_fmt)}))

    elif section.startswith("③"):
        st.subheader("Removing a D-GCAN component barely changes the result")
        d = pd.read_csv(RESULTS / "ablation_v13.csv")
        names = {"full": "Full D-GCAN", "no_gcn": "No graph convolution", "no_gat": "No attention block",
                 "uniform_attn": "Attention → plain average", "no_fp": "No fingerprints"}
        d["Variant"] = d.variant.map(names)
        ch = alt.Chart(d).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Variant:N", sort=list(names.values()), title=None, axis=alt.Axis(labelAngle=-20)),
            column=alt.Column("dataset:N", title=None),
            y=alt.Y("test_auc_mean:Q", scale=alt.Scale(domain=[0.5, 0.9], clamp=True), title="Test AUC"),
            color=alt.condition(alt.datum.variant == "full", alt.value(CORAL), alt.value(INDIGO)),
            tooltip=["Variant", alt.Tooltip("test_auc_mean:Q", format=".3f"), alt.Tooltip("welch_p:Q", format=".3f")]
        ).properties(width=300, height=280)
        st.altair_chart(ch)
        card("""<p style="color:var(--ink)">Only one change is significant, and it <b>improves</b> the model:
        plain atom types instead of D-GCAN's fingerprints on BBBP (0.640 → 0.695, p = 0.0001).</p>""")

    elif section.startswith("④"):
        st.subheader("Re-testing the D-GCAN paper on its own drug-likeness data")
        c1, c2 = st.columns(2)
        with c1:
            st.html("""<div class="claim yes"><b>✓ Graph convolution helps (paper: +6.1%)</b><br>
            <span>Confirmed. Removing it lowers AUC on both splits (p &lt; 0.01).</span></div>""")
        with c2:
            st.html("""<div class="claim no"><b>✗ Attention helps (paper: +4.0%)</b><br>
            <span>Not confirmed. No AUC difference on either split.</span></div>""")
        d = pd.read_csv(RESULTS / "paper_ablation_v16.csv")
        d["lo"], d["hi"] = d.test_auc_mean - d.test_auc_std, d.test_auc_mean + d.test_auc_std
        d["Variant"] = d.variant.map({"full": "Full D-GCAN", "no_gat": "No attention", "no_gcn": "No convolution"})
        base = alt.Chart(d).encode(x=alt.X("Variant:N", title=None, sort=["Full D-GCAN", "No attention", "No convolution"],
                                           axis=alt.Axis(labelAngle=0)))
        bars = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=60).encode(
            y=alt.Y("test_auc_mean:Q", scale=alt.Scale(domain=[0.85, 0.97], clamp=True), title="Test AUC"),
            color=alt.condition(alt.datum.variant == "no_gcn", alt.value(CORAL), alt.value(INDIGO)),
            tooltip=["split", "paper_name", alt.Tooltip("test_auc_mean:Q", format=".3f"), alt.Tooltip("welch_p:Q", format=".3f")])
        err = base.mark_rule(color=PAL["muted"]).encode(y="lo:Q", y2="hi:Q")
        st.altair_chart((bars + err).properties(width=330, height=280).facet(column=alt.Column("split:N", title=None)))
        l, r = st.columns(2)
        with l:
            v = pd.read_csv(RESULTS / "druglike_v15_runs.csv").groupby("mode")[["acc", "auc_hardlabel", "auc_score"]].agg(["mean", "std"])
            card("<h4>Their exact protocol: original vs fixed code</h4>")
            table(pd.DataFrame({"Code": ["Original (bug)", "Fixed"],
                                "Accuracy": [fmt_pm(v.loc[m, ("acc", "mean")], v.loc[m, ("acc", "std")]) for m in ("bug", "fix")],
                                "Their 'AUC'": [f"{v.loc[m, ('auc_hardlabel', 'mean')]:.3f}" for m in ("bug", "fix")],
                                "Real AUC": [fmt_pm(v.loc[m, ("auc_score", "mean")], v.loc[m, ("auc_score", "std")]) for m in ("bug", "fix")]}))
        with r:
            card("""<h4>What this means</h4><p>Paper: accuracy 0.923, "AUC" 0.951. On this balanced dataset the
            bug is <b>harmless</b> — their numbers reproduce. It only breaks training on data like BBBP: a
            hidden, data-dependent bug.</p><p style="margin-top:10px">Also found: no validation set, a single
            run, and "AUC" computed from 0/1 predicted labels.</p>""")

    else:
        st.subheader("Can we trust the explanations? Delete each method's top-3 atoms")
        a = pd.read_csv(RESULTS / "explain_v17_summary.csv")
        b = pd.read_csv(RESULTS / "explain_v18_summary.csv")
        d = pd.concat([a, b[b.method == "shap"]])
        d = d[d.method != "random"].merge(a[a.method == "random"][["dataset", "mean_abs_dp_topk"]],
                                          on="dataset", suffixes=("", "_random"))
        d["ratio"] = d.mean_abs_dp_topk / d.mean_abs_dp_topk_random
        d["Method"] = d.method.map({"occlusion": "Occlusion", "shap": "SHAP", "gnnexplainer": "GNNExplainer",
                                    "attention": "D-GCAN attention"})
        d["Model"] = d.dataset.map({"BBBP": "BBBP", "BACE": "BACE", "druglikeRandom": "Drug-likeness (random)",
                                    "druglikeScaffold": "Drug-likeness (scaffold)"})
        c1, c2 = st.columns([3, 2])
        with c1:
            ch = alt.Chart(d).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
                y=alt.Y("Method:N", sort=["Occlusion", "SHAP", "GNNExplainer", "D-GCAN attention"], title=None),
                x=alt.X("ratio:Q", title="Prediction change vs deleting 3 random atoms (×)"),
                color=alt.condition(alt.datum.method == "attention", alt.value(CORAL), alt.value(INDIGO)),
                row=alt.Row("Model:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
                tooltip=["Model", "Method", alt.Tooltip("ratio:Q", format=".1f")]).properties(width=380, height=90)
            st.altair_chart(ch)
        with c2:
            card("""<h4>Reading</h4><p>1× means the method is no better than picking random atoms.
            D-GCAN's own attention weights sit around 1× — <b>not a faithful explanation</b>. Occlusion and
            SHAP move the prediction 2–7× more.</p><p style="margin-top:10px">Scaffold enrichment ≈ 1 for the
            faithful methods: no sign the model just memorises core structures.</p>""")
        st.image(str(RESULTS / "explain_v18_BBBP.png"), width="stretch")
        st.caption("SHAP on the brain-penetration model. Sucrose: every hydroxyl group (blue) argues against "
                   "crossing into the brain — the chemically expected answer.")

# ------------------------------------------------------------------ how it works
with tab_how:
    stages = [
        ("1", "Molecule → graph", "RDKit reads the SMILES text; atoms become nodes and bonds become edges, "
         "each atom labelled with its local chemical environment."),
        ("2", "Convolution + attention", "Graph convolution passes messages between bonded atoms; graph "
         "attention weights neighbours; the atom vectors are summed into one molecule vector."),
        ("3", "Prediction", "A 10-layer network turns the molecule vector into a probability. Training "
         "uses cross-entropy on the raw scores (our fix).")]
    cards([f'<span class="g-step">Stage {k}</span><h4>{t}</h4><p>{d}</p>' for k, t, d in stages], "c3")
    st.html('<div class="g-section">The bug we found and fixed</div>')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original D-GCAN** — sigmoid, then cross-entropy (double squashing)")
        st.code("scores = torch.sigmoid(self.W_property(vectors))\n"
                "loss = F.cross_entropy(scores, labels)  # softmax again\n"
                "# saturates to 0/1 -> gradients ~1e-10 -> stops learning", language="python")
    with c2:
        st.markdown("**Our fix** — raw logits into cross-entropy")
        st.code("logits = self.W_property(vectors)\n"
                "loss = F.cross_entropy(logits, labels)  # correct\n"
                "probs = F.softmax(logits, dim=1)", language="python")
    st.caption("Applied by replacing one method on the model object; the original library is left untouched.")
    st.html('<div class="g-section">Design diagrams</div>')
    diagram = st.segmented_control("Diagram", ["Use case", "Sequence", "State chart", "Deployment", "Class"],
                                   default="Sequence", key="diagram") or "Sequence"
    files = {"Use case": "4.2.1_Use_Case_Diagram.png", "Sequence": "4.2.2_Sequence_Diagram.png",
             "State chart": "4.2.3_State_Chart_Diagram.png", "Deployment": "4.2.4_Deployment_Diagram.png",
             "Class": "4.2.5_Class_Diagram.png"}
    st.image(str(DIAGRAMS / files[diagram]), width="stretch")

# ------------------------------------------------------------------ verify
with tab_check:
    card("""<h4>Is this the real trained model?</h4><p>Re-scores the entire held-out test set from scratch
    with the saved checkpoint and compares the AUC with the one in our results files.</p>""")
    which = st.segmented_control("Model", list(TASKS), default="BBBP", key="verify_ds",
                                 format_func=lambda k: TASKS[k]["short"]) or "BBBP"
    if st.button("Run check", type="primary"):
        with st.spinner("Scoring the test set…"):
            v = get_predictor(which).verify()
        c1, c2, c3 = st.columns(3)
        c1.metric("Test molecules", v["n"])
        c2.metric("AUC — computed now", f"{v['auc_now']:.4f}")
        c3.metric("AUC — reported", f"{v['auc_saved']:.4f}")
        if abs(v["auc_now"] - v["auc_saved"]) < 1e-6:
            st.success(f"Match ✓ — largest per-molecule difference {v['max_diff']:.1e}")
        else:
            st.error("Mismatch")

st.html(f'<div class="footer">Graph2Drug · <a href="{GITHUB}" target="_blank">GitHub</a></div>')
