"""
v18 — SHAP (Shapley values) for D-GCAN atoms (analysis only, no training).

Adds SHAP (Lundberg & Lee, NeurIPS 2017) to the v14/v17 comparison of occlusion,
GNNExplainer and GAT attention.

Shapley game: the players are the molecule's atoms (explicit H included, as the model
sees them). A coalition keeps its atoms; an absent atom has its fingerprint embedding
zeroed — the same removal operation as occlusion and the GNNExplainer mask, so the four
methods differ only in how they attribute, not in what "removing an atom" means.
Value = predicted probability of the positive class. Background = all atoms removed.
Estimated with shap.KernelExplainer (nsamples = 2M + 2048, the library's "auto"
setting). Efficiency check: sum of SHAP values = p(molecule) - p(no atoms).

Unlike single-atom occlusion, SHAP averages an atom's effect over many contexts, so it
also credits atoms that only matter together (e.g. one of two redundant hydroxyls,
which occlusion scores as ~0). The rank correlation with occlusion measures how much
that matters for D-GCAN.

Same evaluation as v17, on the same seed-42 checkpoints:
  1. Faithfulness: zero the top-K atoms by |SHAP| and measure |Δp| vs. K random atoms
     (occlusion recomputed here as the reference).
  2. Scaffold enrichment of the top-K heavy atoms (1 = no scaffold preference).
  3. Case-study figure (signed SHAP: red supports the positive class, blue opposes).

Runs locally on CPU. Env: DATASETS (comma list), MAX_MOLS (smoke tests), NSAMPLES.
Outputs: results/explain_v18_{faithfulness,scaffold,summary}.csv, explain_v18_{DATASET}.png
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import demo_predict as dp  # noqa: E402

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
K = 3
N_RANDOM = 20
MAX_MOLS = int(os.environ.get("MAX_MOLS", 0))  # 0 = whole test set
NSAMPLES = os.environ.get("NSAMPLES", "auto")
rng = np.random.RandomState(0)
np.random.seed(0)  # KernelExplainer samples coalitions with numpy's global RNG

for name in ("druglikeRandom", "druglikeScaffold"):
    ckpt = MODELS / f"{name}_v16_full_seed42_best.pth"
    if ckpt.exists():
        dp.DATASETS[name] = {"checkpoint": ckpt,
                             "saved_predictions": RESULTS / f"{name}_v16_full_seed42_test_predictions.csv",
                             "positive": "drug-like", "negative": "not drug-like"}
DATASETS = [d for d in os.environ.get("DATASETS", "BBBP,BACE,druglikeRandom,druglikeScaffold").split(",")
            if d in dp.DATASETS]

CASES = {
    "BBBP": {"Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
             "Diazepam": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3",
             "Penicillin G": "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O",
             "Sucrose": "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O"},
    "BACE": {"Inhibitor A": "Cc1cc(C2(c3cccc(C#CC4CC4)c3)N=C(N)c3c(F)cccc32)cn(C)c1=O",
             "Inhibitor B": "CN(C(=O)CCc1cc2ccccc2nc1N)C1CCCCC1",
             "Non-inhibitor A": "Nc1nc(CCc2ccc3cc[nH]c3c2)cc(=O)[nH]1",
             "Non-inhibitor B": "CC[C@@H](CC(=O)NC1C2CC3CC(C2)CC1C3)n1c(N)nc2cc(Cl)ccc21"},
}


@torch.no_grad()
def prob(model, vectors, adj):
    for l in range(model.layer_hidden):
        vectors = torch.nn.functional.normalize(model.update(adj, vectors, l), 2, 1)
    mol = model.attentions(vectors, adj).sum(0, keepdim=True)
    for l in range(model.layer_output):
        mol = torch.relu(model.W_output[l](mol))
    return torch.softmax(model.W_property(mol)[0], 0)[1].item()


def delete(model, vecs, adj, idx):
    v = vecs.clone(); v[list(idx)] = 0
    return prob(model, v, adj)


def explain(model, item):
    _, fps, adj, _, _ = item
    adj = adj.float()
    with torch.no_grad():
        vecs = model.embed_fingerprint(fps)
    n = len(fps)
    p0 = prob(model, vecs, adj)
    occl = np.array([p0 - delete(model, vecs, adj, [i]) for i in range(n)])

    def f(Z):  # Z: coalitions (rows) x atoms, 1 = atom kept
        z = torch.as_tensor(Z, dtype=vecs.dtype)
        return np.array([prob(model, vecs * row[:, None], adj) for row in z])

    explainer = shap.KernelExplainer(f, np.zeros((1, n)))
    sv = np.asarray(explainer.shap_values(np.ones((1, n)), nsamples=NSAMPLES, silent=True)).reshape(-1)
    efficiency_err = abs(sv.sum() - (p0 - float(explainer.expected_value)))
    return p0, sv, occl, vecs, adj, efficiency_err


def scaffold_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    core = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
    if core.GetNumAtoms() == 0:
        return mol.GetNumAtoms(), set()
    return mol.GetNumAtoms(), set(mol.GetSubstructMatch(core))


from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402  (registers Chem.Scaffolds)

fair_rows, scaf_rows = [], []
for name in DATASETS:
    pred = dp.Predictor(name)
    model = pred.model
    items = pred.test_set[:MAX_MOLS] if MAX_MOLS else pred.test_set
    print(f"\n=== {name}: {len(items)} test molecules ===", flush=True)
    t0 = time.time()
    for j, item in enumerate(items):
        smi = item[0]
        n_heavy, scaf = scaffold_atoms(smi)
        if len(item[1]) <= K or n_heavy <= K:
            continue
        p0, sv, occl, vecs, adj, eff = explain(model, item)
        scores = {"shap": np.abs(sv), "occlusion": np.abs(occl)}
        row = {"dataset": name, "smiles": smi, "p": p0, "n_atoms": len(sv), "efficiency_err": eff,
               "spearman_shap_occlusion": spearmanr(sv, occl).correlation}
        for meth, sc in scores.items():
            row[f"dp_{meth}_topk"] = abs(p0 - delete(model, vecs, adj, np.argsort(-sc)[:K]))
        row["dp_random_k"] = np.mean([abs(p0 - delete(model, vecs, adj, rng.choice(len(vecs), K, replace=False)))
                                      for _ in range(N_RANDOM)])
        fair_rows.append(row)
        if scaf and len(scaf) < n_heavy:
            base = len(scaf) / n_heavy
            srow = {"dataset": name, "smiles": smi, "label": int(item[-1].item()), "scaffold_frac": base}
            for meth, sc in scores.items():
                srow[f"enrich_{meth}"] = np.mean([i in scaf for i in np.argsort(-sc[:n_heavy])[:K]]) / base
            scaf_rows.append(srow)
        if (j + 1) % 25 == 0:
            print(f"  {j + 1}/{len(items)} molecules, {(time.time() - t0) / 60:.1f} min", flush=True)
    f = pd.DataFrame([r for r in fair_rows if r["dataset"] == name])
    s = pd.DataFrame([r for r in scaf_rows if r["dataset"] == name])
    for meth in ("shap", "occlusion"):
        print(f"{meth:10s} |Δp| top-{K} = {f[f'dp_{meth}_topk'].mean():.4f} "
              f"(beats random on {100 * (f[f'dp_{meth}_topk'] > f.dp_random_k).mean():.0f}%) | "
              f"scaffold enrichment = {s[f'enrich_{meth}'].mean():.2f}", flush=True)
    print(f"random     |Δp| = {f.dp_random_k.mean():.4f} | Spearman(SHAP, occlusion) median "
          f"{f.spearman_shap_occlusion.median():.2f} | max efficiency error {f.efficiency_err.max():.1e} | "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)

    if name in CASES and not MAX_MOLS:
        feats = pred.featurise(list(CASES[name].values()))
        tiles = []
        for label, smi in CASES[name].items():
            p0, sv, _, _, _, _ = explain(model, feats[smi])
            mol = Chem.MolFromSmiles(smi)
            imp = sv[:mol.GetNumAtoms()]
            scale = max(np.abs(imp).max(), 1e-9)
            colors = {i: ((1.0, float(1 - min(v / scale, 1)), float(1 - min(v / scale, 1))) if v > 0 else
                          (float(1 - min(-v / scale, 1)), float(1 - min(-v / scale, 1)), 1.0))
                      for i, v in enumerate(imp)}
            d = rdMolDraw2D.MolDraw2DCairo(420, 320)
            d.drawOptions().legendFontSize = 18
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(colors), highlightAtomColors=colors,
                                               highlightBonds=[], legend=f"{label}  (p = {p0:.2f})")
            d.FinishDrawing()
            tiles.append(Image.open(io.BytesIO(d.GetDrawingText())))
            top = np.argsort(-np.abs(imp))[:3]
            print(f"  {label}: top SHAP atoms " + ", ".join(
                f"{mol.GetAtomWithIdx(int(i)).GetSymbol()}{int(i)}({'+' if imp[i] > 0 else '-'})" for i in top))
        w, h = tiles[0].size
        sheet = Image.new("RGB", (w * len(tiles), h), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * w, 0))
        sheet.save(RESULTS / f"explain_v18_{name}.png")

if MAX_MOLS:
    print("\nsmoke test only — nothing saved")
    sys.exit(0)
fair, scaf = pd.DataFrame(fair_rows), pd.DataFrame(scaf_rows)
fair.to_csv(RESULTS / "explain_v18_faithfulness.csv", index=False)
scaf.to_csv(RESULTS / "explain_v18_scaffold.csv", index=False)
summary = []
for name in DATASETS:
    f, s = fair[fair.dataset == name], scaf[scaf.dataset == name]
    for meth in ("shap", "occlusion", "random"):
        col = "dp_random_k" if meth == "random" else f"dp_{meth}_topk"
        summary.append({"dataset": name, "method": meth, "mean_abs_dp_topk": f[col].mean(),
                        "beats_random_pct": np.nan if meth == "random" else 100 * (f[col] > f.dp_random_k).mean(),
                        "scaffold_enrichment": np.nan if meth == "random" else s[f"enrich_{meth}"].mean(),
                        "spearman_shap_occlusion_median": f.spearman_shap_occlusion.median(),
                        "n_molecules": len(f)})
pd.DataFrame(summary).to_csv(RESULTS / "explain_v18_summary.csv", index=False)
print("\n" + pd.DataFrame(summary).round(3).to_string(index=False))
print("V18 DONE")
