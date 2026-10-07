"""
v14 — explainability case studies (roadmap step 6), analysis only (no training).

For a trained, fixed D-GCAN checkpoint, scores every atom of a molecule two ways:
  occlusion  — zero that atom's fingerprint embedding and re-run the model; the
               drop in predicted probability is the atom's importance (model-agnostic;
               GNNExplainer needs a PyTorch Geometric model, D-GCAN is not one).
  attention  — mean attention the atom *receives* across the GAT heads (first GAT
               layer), D-GCAN's own built-in signal.

Faithfulness check (whole test set): delete (zero) the top-k atoms ranked by each
method and compare the probability change with deleting k random atoms. A faithful
ranking should move the prediction more than random.

Case studies: well-known drugs for BBBP and held-out test molecules for BACE, drawn
with atoms coloured by occlusion importance.

Runs locally on CPU. Uses the BBBP v5 (= v8 seed 42) and BACE v9 seed-42 checkpoints
via experiments/demo_predict.Predictor, which rebuilds each model's exact vocabulary.
Outputs: results/explain_v14_faithfulness.csv, results/explain_v14_cases.csv,
         results/explain_v14_{BBBP,BACE}.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import demo_predict as dp  # noqa: E402

RESULTS = ROOT / "results"
K = 3          # atoms deleted in the faithfulness test
N_RANDOM = 20  # random-deletion repeats per molecule
rng = np.random.RandomState(0)

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


# ---------------------------------------------------------------- model pieces
@torch.no_grad()
def prob_from_vectors(model, vectors, adj):
    """D-GCAN forward for ONE molecule, starting from (possibly edited) fingerprint
    embeddings: GCN layers -> GAT -> sum -> MLP -> softmax (the fixed, logits path)."""
    for l in range(model.layer_hidden):
        vectors = F.normalize(model.update(adj, vectors, l), 2, 1)
    atom_out = model.attentions(vectors, adj)
    mol = atom_out.sum(0, keepdim=True)
    for l in range(model.layer_output):
        mol = torch.relu(model.W_output[l](mol))
    return torch.softmax(model.W_property(mol), dim=1)[0, 1].item()


@torch.no_grad()
def attention_received(model, vectors, adj):
    """Mean attention each atom receives in the first GAT layer, averaged over heads."""
    for l in range(model.layer_hidden):
        vectors = F.normalize(model.update(adj, vectors, l), 2, 1)
    scores = []
    for layer in model.attentions.attentions:
        h = vectors @ layer.W
        n = h.shape[0]
        a_in = torch.cat([h.repeat(1, n).view(n * n, -1), h.repeat(n, 1)], dim=1).view(n, n, -1)
        e = layer.leakyrelu((a_in @ layer.a).squeeze(2))
        att = torch.softmax(torch.where(adj > 0, e, torch.full_like(e, -9e10)), dim=1)
        scores.append(att.mean(0))  # column mean = attention received
    return torch.stack(scores).mean(0).numpy()


def explain(model, item):
    _, fps, adj, _, _ = item
    adj = adj.float()
    base_vecs = model.embed_fingerprint(fps)
    p0 = prob_from_vectors(model, base_vecs, adj)
    occl = np.zeros(len(fps))
    for i in range(len(fps)):
        v = base_vecs.clone()
        v[i] = 0
        occl[i] = p0 - prob_from_vectors(model, v, adj)
    return p0, occl, attention_received(model, base_vecs, adj), base_vecs, adj


def delete(model, vecs, adj, idx):
    v = vecs.clone()
    v[list(idx)] = 0
    return prob_from_vectors(model, v, adj)


# ---------------------------------------------------------------- run
fair_rows, case_rows = [], []
for name, cases in CASES.items():
    pred = dp.Predictor(name)
    model = pred.model
    print(f"\n=== {name}: faithfulness on {len(pred.test_set)} test molecules (k={K}) ===")
    for item in pred.test_set:
        if len(item[1]) <= K:
            continue
        p0, occl, att, vecs, adj = explain(model, item)
        # deletion should push the prediction towards the opposite class: measure |Δp|
        d_occl = abs(p0 - delete(model, vecs, adj, np.argsort(-np.abs(occl))[:K]))
        d_att = abs(p0 - delete(model, vecs, adj, np.argsort(-att)[:K]))
        d_rand = np.mean([abs(p0 - delete(model, vecs, adj, rng.choice(len(occl), K, replace=False)))
                          for _ in range(N_RANDOM)])
        fair_rows.append({"dataset": name, "n_atoms": len(occl), "p": p0,
                          "dp_occlusion_topk": d_occl, "dp_attention_topk": d_att, "dp_random_k": d_rand})
    f = pd.DataFrame([r for r in fair_rows if r["dataset"] == name])
    print(f"mean |Δp| deleting top-{K}: occlusion {f.dp_occlusion_topk.mean():.4f} | "
          f"attention {f.dp_attention_topk.mean():.4f} | random {f.dp_random_k.mean():.4f}")
    print(f"occlusion beats random on {100 * (f.dp_occlusion_topk > f.dp_random_k).mean():.0f}% of molecules; "
          f"attention on {100 * (f.dp_attention_topk > f.dp_random_k).mean():.0f}%")

    # case-study figure
    feats = pred.featurise(list(cases.values()))
    imgs, legends = [], []
    for label, smi in cases.items():
        p0, occl, att, _, _ = explain(model, feats[smi])
        mol = Chem.MolFromSmiles(smi)
        # preprocess adds explicit Hs (Chem.AddHs); heavy atoms keep their indices first
        heavy = mol.GetNumAtoms()
        imp = occl[:heavy]
        scale = max(np.abs(imp).max(), 1e-9)
        colors = {i: ((1.0, 1 - min(v / scale, 1), 1 - min(v / scale, 1)) if v > 0 else
                      (1 - min(-v / scale, 1), 1 - min(-v / scale, 1), 1.0)) for i, v in enumerate(imp)}
        d = rdMolDraw2D.MolDraw2DCairo(420, 320)
        d.drawOptions().legendFontSize = 18
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(colors), highlightAtomColors=colors,
                                           highlightBonds=[], legend=f"{label}  (p = {p0:.2f})")
        d.FinishDrawing()
        imgs.append(d.GetDrawingText())
        top = np.argsort(-np.abs(imp))[:3]  # strongest effect either way; sign: + supports positive
        case_rows.append({"dataset": name, "molecule": label, "smiles": smi, "p": p0,
                          "top_atoms_occlusion": ";".join(f"{mol.GetAtomWithIdx(int(i)).GetSymbol()}{int(i)}"
                                                          f"({'+' if imp[i] > 0 else '-'})" for i in top),
                          "top_atoms_attention": ";".join(f"{mol.GetAtomWithIdx(int(i)).GetSymbol()}{int(i)}"
                                                          for i in np.argsort(-att[:heavy])[:3])})
    # stitch PNGs
    from PIL import Image
    import io
    tiles = [Image.open(io.BytesIO(b)) for b in imgs]
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * len(tiles), h), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * w, 0))
    sheet.save(RESULTS / f"explain_v14_{name}.png")

pd.DataFrame(fair_rows).to_csv(RESULTS / "explain_v14_faithfulness.csv", index=False)
pd.DataFrame(case_rows).to_csv(RESULTS / "explain_v14_cases.csv", index=False)
print("\n" + pd.DataFrame(case_rows)[["dataset", "molecule", "p", "top_atoms_occlusion", "top_atoms_attention"]].to_string(index=False))
print("\nRed = atom supports the positive prediction (removing it lowers p); blue = opposes it.")
