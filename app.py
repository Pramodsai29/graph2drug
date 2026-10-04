"""
Graph2Drug — web demo.

    streamlit run app.py

Runs the trained D-GCAN checkpoints locally on CPU (no GPU or internet needed).
"""
import statistics
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
import demo_predict as dp  # noqa: E402

RESULTS = ROOT / "results"
DIAGRAMS = ROOT / "docs" / "diagrams"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"

TASKS = {
    "BBBP": {
        "title": "Blood–Brain Barrier Penetration",
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
        "title": "BACE-1 Inhibition (Alzheimer's disease)",
        "question": "Does this molecule inhibit BACE-1, an enzyme linked to Alzheimer's disease?",
        "examples": {
            "Test-set inhibitor A": "Cc1cc(C2(c3cccc(C#CC4CC4)c3)N=C(N)c3c(F)cccc32)cn(C)c1=O",
            "Test-set inhibitor B": "CN(C(=O)CCc1cc2ccccc2nc1N)C1CCCCC1",
            "Test-set non-inhibitor A": "Nc1nc(CCc2ccc3cc[nH]c3c2)cc(=O)[nH]1",
            "Test-set non-inhibitor B": "CC[C@@H](CC(=O)NC1C2CC3CC(C2)CC1C3)n1c(N)nc2cc(Cl)ccc21",
        },
    },
}

st.set_page_config(page_title="Graph2Drug", page_icon="🧪", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained model…")
def get_predictor(name):
    return dp.Predictor(name)


def molecule_card(mol):
    st.image(Draw.MolToImage(mol, size=(380, 260)), width=380)
    st.markdown(f"**Formula** {rdMolDescriptors.CalcMolFormula(mol)} &nbsp;·&nbsp; "
                f"**Mol. weight** {Descriptors.MolWt(mol):.1f} &nbsp;·&nbsp; "
                f"**Heavy atoms** {mol.GetNumHeavyAtoms()}")


def result_card(pred, p):
    positive = p > dp.THRESHOLD
    v = pred.verdict(p)
    (st.success if positive else st.error)(f"### {v[0].upper() + v[1:]}")
    st.metric("Predicted probability", f"{p:.3f}")
    st.progress(min(max(p, 0.0), 1.0))
    st.caption(f"Positive if probability > {dp.THRESHOLD} (the threshold used in the "
               "original D-GCAN evaluation).")


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🧪 Graph2Drug")
    st.write("Molecular property prediction with a corrected **D-GCAN** "
             "(Directed Graph Convolutional Attention Network).")
    st.divider()
    st.markdown("**Team**  \nPramod Sai Tavva  \nReshvanth  \nRahman  \nVarun")
    st.markdown("**Guide**  \nDr. K. Bhargavi")
    st.caption("Department of Information Technology  \nKeshav Memorial Institute of Technology")
    st.divider()
    st.caption("Runs locally on CPU · trained checkpoints from the 140-epoch runs")

st.title("Graph2Drug: Predicting Drug Properties from Molecular Graphs")
tab_pred, tab_how, tab_res, tab_check = st.tabs(
    ["🔬 Predict", "⚙️ How it works", "📊 Results", "✅ Model check"])

# ---------------- Predict ----------------
with tab_pred:
    task = st.radio("Prediction task", list(TASKS), horizontal=True,
                    format_func=lambda k: f"{k} — {TASKS[k]['title']}")
    cfg = TASKS[task]
    st.write(f"**{cfg['question']}**")

    left, right = st.columns([2, 1])
    with left:
        choice = st.selectbox("Pick an example molecule, or type your own SMILES below",
                              ["— custom —"] + list(cfg["examples"]), index=1, key=f"ex_{task}")
        default = cfg["examples"].get(choice, "")
        smiles = st.text_input("SMILES", value=default, key=f"smi_{task}_{choice}",
                               help="Text notation for a molecule — copy one from PubChem")
    with right:
        st.write("")
        st.write("")
        go = st.button("Predict", type="primary", width="stretch")

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.warning("That isn't a valid SMILES string — RDKit could not parse it.")
        else:
            pred = get_predictor(task)
            p = pred.predict([smiles])[smiles]
            a, b = st.columns([1, 1])
            with a:
                molecule_card(mol)
            with b:
                result_card(pred, p)

    with st.expander(f"Compare all {task} example molecules"):
        pred = get_predictor(task)
        probs = pred.predict(list(cfg["examples"].values()))
        st.dataframe(pd.DataFrame([
            {"Molecule": n, "Probability": round(probs[s], 4), "Prediction": pred.verdict(probs[s])}
            for n, s in cfg["examples"].items()]), hide_index=True, width="stretch")

# ---------------- How it works ----------------
with tab_how:
    st.subheader("System architecture")
    st.image(str(DIAGRAMS / "1.5_Architecture_Diagram.png"), width="stretch")
    st.markdown(
        "1. **Molecule → graph.** Atoms become nodes and bonds become edges (RDKit).\n"
        "2. **Graph convolution + attention.** D-GCAN passes messages between neighbouring "
        "atoms and uses attention to weight the important ones.\n"
        "3. **Prediction.** The pooled molecule vector goes through an MLP to a probability.")
    st.subheader("The bug we found and fixed")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original D-GCAN** — sigmoid, then cross-entropy (double squashing)")
        st.code("scores = torch.sigmoid(self.W_property(vectors))\n"
                "loss = F.cross_entropy(scores, labels)   # softmax applied again\n"
                "# sigmoid saturates to 0/1 -> gradients ~1e-10 -> model stops learning",
                language="python")
    with c2:
        st.markdown("**Our fix** — raw logits into cross-entropy")
        st.code("logits = self.W_property(vectors)\n"
                "loss = F.cross_entropy(logits, labels)   # correct\n"
                "probs = F.softmax(logits, dim=1)",
                language="python")
    st.caption("Applied by replacing one method on the model object — the original library "
               "is left untouched (see experiments/train_dataset_v8.py).")

# ---------------- Results ----------------
with tab_res:
    def mean_std(path):
        v = pd.read_csv(path)["test_auc"]
        return v.mean(), v.std()

    rows = [("BBBP", *mean_std(RESULTS / "BBBP_v8_multiseed_results.csv")),
            ("BACE", *mean_std(RESULTS / "BACE_v9_multiseed_results.csv")),
            ("ClinTox", *mean_std(RESULTS / "ClinTox_v9_multiseed_results.csv")),
            ("Tox21", *mean_std(RESULTS / "Tox21_v9_multiseed_results.csv"))]
    res = pd.DataFrame(rows, columns=["Dataset", "mean", "std"])
    res["lo"], res["hi"] = res["mean"] - res["std"], res["mean"] + res["std"]
    res["label"] = res.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)

    c1, *fixed_cols = st.columns(1 + len(rows))
    c1.metric("Original D-GCAN (BBBP)", "0.466", help="Test ROC-AUC before the fix")
    for col, r in zip(fixed_cols, rows):
        col.metric(f"Fixed — {r[0]}", f"{r[1]:.3f} ± {r[2]:.3f}")
    st.caption("Test ROC-AUC on scaffold splits, mean ± std over 5 training seeds. "
               "0.5 = random guessing, 1.0 = perfect.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Test AUC by dataset (fixed model)**")
        base = alt.Chart(res).encode(x=alt.X("Dataset:N", sort=None, title=None,
                                             axis=alt.Axis(labelAngle=0, labelFontSize=13)))
        bars = base.mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=48).encode(
            y=alt.Y("mean:Q", scale=alt.Scale(domain=[0, 1]), title="Test ROC-AUC"),
            tooltip=["Dataset", alt.Tooltip("label:N", title="AUC")])
        err = base.mark_rule(color="#333").encode(y="lo:Q", y2="hi:Q")
        text = base.mark_text(dy=-14, color="#222").encode(y="hi:Q", text="label:N")
        chance = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color=GRAY).encode(y="y:Q")
        st.altair_chart((bars + err + text + chance).properties(height=320), width="stretch")
    with right:
        st.markdown("**BBBP validation AUC while training: original vs fixed**")
        v1 = pd.read_csv(RESULTS / "BBBP_training_log.csv")[["epoch", "valid_auc"]].assign(Model="Original")
        v5 = pd.read_csv(RESULTS / "BBBP_v5_training_log.csv")[["epoch", "valid_auc"]].assign(Model="Fixed")
        curves = pd.concat([v1, v5])
        line = alt.Chart(curves).mark_line(strokeWidth=2).encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("valid_auc:Q", title="Validation AUC", scale=alt.Scale(domain=[0.2, 1])),
            color=alt.Color("Model:N", scale=alt.Scale(domain=["Fixed", "Original"], range=[BLUE, ORANGE]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Model", "epoch", alt.Tooltip("valid_auc:Q", format=".3f")])
        st.altair_chart(line.properties(height=320), width="stretch")

    st.markdown("**Original model's output (before the fix)** — every test molecule scored ≈ 1.0")
    st.image(str(RESULTS / "BBBP_v2_score_histogram.png"), width=520)

# ---------------- Model check ----------------
with tab_check:
    st.write("Re-scores the entire held-out test set from scratch with the saved checkpoint "
             "and compares against the AUC reported in our results files — showing the demo "
             "uses the real trained model.")
    which = st.radio("Dataset", list(TASKS), horizontal=True, key="verify_ds")
    if st.button("Run check", type="primary"):
        with st.spinner("Scoring test set…"):
            v = get_predictor(which).verify()
        c1, c2, c3 = st.columns(3)
        c1.metric("Test molecules", v["n"])
        c2.metric("AUC — computed now", f"{v['auc_now']:.4f}")
        c3.metric("AUC — reported", f"{v['auc_saved']:.4f}")
        if abs(v["auc_now"] - v["auc_saved"]) < 1e-6:
            st.success(f"Match ✓ (largest per-molecule difference {v['max_diff']:.1e})")
        else:
            st.error("Mismatch")
