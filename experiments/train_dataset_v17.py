"""
v17 — GNNExplainer for D-GCAN + scaffold-shortcut analysis (analysis only, no training).

Extends v14 (occlusion vs. GAT attention) with GNNExplainer (Ying et al., NeurIPS 2019)
and asks whether explanations concentrate on Bemis–Murcko scaffold atoms.

GNNExplainer variant: node ("object") mask. D-GCAN's GAT layer only tests whether a
bond exists (adj > 0), so a soft edge mask would be silently ignored there; instead a
mask m_v in (0,1) per atom scales that atom's input fingerprint embedding (mask 0 =
the atom removed, exactly as occlusion removes it). Optimised per molecule with Adam:
    loss = -log p(predicted class | masked molecule)
           + SIZE_COEF * mean(m) + ENT_COEF * mean(binary entropy(m))
(GNNExplainer's prediction-preservation + sparsity + entropy terms, with PyTorch
Geometric's default coefficients.) Atom importance = learned mask value.

Outputs for each dataset (BBBP, BACE; plus druglikeRandom / druglikeScaffold once the
v16 full-model checkpoints exist):
  1. Faithfulness (same test as v14): zero the top-K atoms ranked by each method and
     measure |Δp|, vs. K random atoms. Unlike single-atom occlusion, GNNExplainer is
     not optimised for this test, so it is a fairer judge.
  2. Scaffold enrichment: share of each method's top-K heavy atoms that lie on the
     Murcko scaffold, divided by the scaffold's share of the molecule's heavy atoms.
     >1 = explanations favour scaffold atoms (a shortcut signal when the dataset is
     scaffold-separable, as the drug-likeness data is).
  3. Case-study figure of GNNExplainer masks for the same molecules as v14.

Runs locally on CPU. Outputs: results/explain_v17_{faithfulness,scaffold,summary}.csv,
results/explain_v17_{DATASET}.png
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import demo_predict as dp  # noqa: E402

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
K = 3
N_RANDOM = 20
# PyTorch Geometric GNNExplainer defaults for a node mask: 100 epochs, lr 0.01,
# node_feat_size 1.0 (mean reduction), node_feat_ent 0.1, mask init N(0, 0.1) pre-sigmoid
STEPS, LR, SIZE_COEF, ENT_COEF = 100, 0.01, 1.0, 0.1
torch.manual_seed(0)
rng = np.random.RandomState(0)

# drug-likeness models from v16 (registered only if the checkpoints exist)
for name in ("druglikeRandom", "druglikeScaffold"):
    ckpt = MODELS / f"{name}_v16_full_seed42_best.pth"
    if ckpt.exists():
        dp.DATASETS[name] = {"checkpoint": ckpt,
                             "saved_predictions": RESULTS / f"{name}_v16_full_seed42_test_predictions.csv",
                             "positive": "drug-like", "negative": "not drug-like"}
DATASETS = [d for d in ("BBBP", "BACE", "druglikeRandom", "druglikeScaffold") if d in dp.DATASETS]

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


# ---------------------------------------------------------------- model pieces (as v14)
def logits_from_vectors(model, vectors, adj):
    for l in range(model.layer_hidden):
        vectors = F.normalize(model.update(adj, vectors, l), 2, 1)
    mol = model.attentions(vectors, adj).sum(0, keepdim=True)
    for l in range(model.layer_output):
        mol = torch.relu(model.W_output[l](mol))
    return model.W_property(mol)[0]


@torch.no_grad()
def prob(model, vectors, adj):
    return torch.softmax(logits_from_vectors(model, vectors, adj), 0)[1].item()


@torch.no_grad()
def attention_received(model, vectors, adj):
    for l in range(model.layer_hidden):
        vectors = F.normalize(model.update(adj, vectors, l), 2, 1)
    scores = []
    for layer in model.attentions.attentions:
        h = vectors @ layer.W
        n = h.shape[0]
        a_in = torch.cat([h.repeat(1, n).view(n * n, -1), h.repeat(n, 1)], dim=1).view(n, n, -1)
        e = layer.leakyrelu((a_in @ layer.a).squeeze(2))
        att = torch.softmax(torch.where(adj > 0, e, torch.full_like(e, -9e10)), dim=1)
        scores.append(att.mean(0))
    return torch.stack(scores).mean(0).numpy()


def gnnexplainer(model, vectors, adj, target):
    """Node-object-mask GNNExplainer: learn m in (0,1)^n keeping the predicted class."""
    for p in model.parameters():
        p.requires_grad_(False)
    w = (torch.randn(vectors.shape[0], 1) * 0.1).requires_grad_(True)  # sigmoid ≈ 0.5, as PyG
    opt = torch.optim.Adam([w], lr=LR)
    for _ in range(STEPS):
        m = torch.sigmoid(w)
        logp = torch.log_softmax(logits_from_vectors(model, vectors * m, adj), 0)[target]
        ent = -(m * torch.log(m + 1e-8) + (1 - m) * torch.log(1 - m + 1e-8))
        loss = -logp + SIZE_COEF * m.mean() + ENT_COEF * ent.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.sigmoid(w).detach().squeeze(1).numpy()


def explain(model, item):
    _, fps, adj, _, _ = item
    adj = adj.float()
    with torch.no_grad():
        vecs = model.embed_fingerprint(fps)
    p0 = prob(model, vecs, adj)
    occl = np.zeros(len(fps))
    for i in range(len(fps)):
        v = vecs.clone(); v[i] = 0
        occl[i] = p0 - prob(model, v, adj)
    att = attention_received(model, vecs, adj)
    gex = gnnexplainer(model, vecs, adj, int(p0 > 0.5))
    return p0, {"occlusion": np.abs(occl), "gnnexplainer": gex, "attention": att}, occl, vecs, adj


def delete(model, vecs, adj, idx):
    v = vecs.clone(); v[list(idx)] = 0
    return prob(model, v, adj)


def scaffold_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core.GetNumAtoms() == 0:
        return mol.GetNumAtoms(), set()
    match = mol.GetSubstructMatch(core)
    return mol.GetNumAtoms(), set(match)


# ---------------------------------------------------------------- run
fair_rows, scaf_rows = [], []
for name in DATASETS:
    pred = dp.Predictor(name)
    model = pred.model
    print(f"\n=== {name}: {len(pred.test_set)} test molecules ===", flush=True)
    for item in pred.test_set:
        smi = item[0]
        n_heavy, scaf = scaffold_atoms(smi)
        if len(item[1]) <= K or n_heavy <= K:
            continue
        p0, scores, _, vecs, adj = explain(model, item)
        row = {"dataset": name, "smiles": smi, "p": p0}
        for meth, sc in scores.items():
            row[f"dp_{meth}_topk"] = abs(p0 - delete(model, vecs, adj, np.argsort(-sc)[:K]))
        row["dp_random_k"] = np.mean([abs(p0 - delete(model, vecs, adj, rng.choice(len(vecs), K, replace=False)))
                                      for _ in range(N_RANDOM)])
        fair_rows.append(row)
        # scaffold enrichment over heavy atoms only (preprocess appends explicit H after them)
        if scaf and len(scaf) < n_heavy:
            base = len(scaf) / n_heavy
            srow = {"dataset": name, "smiles": smi, "label": int(item[-1].item()), "scaffold_frac": base}
            for meth, sc in scores.items():
                top = np.argsort(-sc[:n_heavy])[:K]
                srow[f"enrich_{meth}"] = (np.mean([i in scaf for i in top])) / base
            scaf_rows.append(srow)
    f = pd.DataFrame([r for r in fair_rows if r["dataset"] == name])
    s = pd.DataFrame([r for r in scaf_rows if r["dataset"] == name])
    for meth in ("occlusion", "gnnexplainer", "attention"):
        print(f"{meth:12s} |Δp| top-{K} = {f[f'dp_{meth}_topk'].mean():.4f} "
              f"(beats random on {100 * (f[f'dp_{meth}_topk'] > f.dp_random_k).mean():.0f}%) | "
              f"scaffold enrichment = {s[f'enrich_{meth}'].mean():.2f}", flush=True)
    print(f"random       |Δp| = {f.dp_random_k.mean():.4f} | mean scaffold share of molecule = {s.scaffold_frac.mean():.2f}")

    # case-study figure (GNNExplainer masks) for the v14 molecules
    if name in CASES:
        feats = pred.featurise(list(CASES[name].values()))
        tiles = []
        for label, smi in CASES[name].items():
            p0, scores, _, _, _ = explain(model, feats[smi])
            mol = Chem.MolFromSmiles(smi)
            imp = scores["gnnexplainer"][:mol.GetNumAtoms()]
            imp = (imp - imp.min()) / max(imp.max() - imp.min(), 1e-9)  # per-molecule rescale for display
            colors = {i: (1.0, float(1 - v), float(1 - v)) for i, v in enumerate(imp)}
            d = rdMolDraw2D.MolDraw2DCairo(420, 320)
            d.drawOptions().legendFontSize = 18
            rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(colors), highlightAtomColors=colors,
                                               highlightBonds=[], legend=f"{label}  (p = {p0:.2f})")
            d.FinishDrawing()
            tiles.append(Image.open(io.BytesIO(d.GetDrawingText())))
        w, h = tiles[0].size
        sheet = Image.new("RGB", (w * len(tiles), h), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * w, 0))
        sheet.save(RESULTS / f"explain_v17_{name}.png")

fair, scaf = pd.DataFrame(fair_rows), pd.DataFrame(scaf_rows)
fair.to_csv(RESULTS / "explain_v17_faithfulness.csv", index=False)
scaf.to_csv(RESULTS / "explain_v17_scaffold.csv", index=False)
summary = []
for name in DATASETS:
    f, s = fair[fair.dataset == name], scaf[scaf.dataset == name]
    for meth in ("occlusion", "gnnexplainer", "attention", "random"):
        col = "dp_random_k" if meth == "random" else f"dp_{meth}_topk"
        summary.append({"dataset": name, "method": meth, "mean_abs_dp_topk": f[col].mean(),
                        "beats_random_pct": np.nan if meth == "random" else 100 * (f[col] > f.dp_random_k).mean(),
                        "scaffold_enrichment": np.nan if meth == "random" else s[f"enrich_{meth}"].mean(),
                        "n_molecules": len(f)})
pd.DataFrame(summary).to_csv(RESULTS / "explain_v17_summary.csv", index=False)
print("\n" + pd.DataFrame(summary).round(3).to_string(index=False))
print("V17 DONE")
