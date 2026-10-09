"""
Per-atom explanations for one molecule, for the web demo (app.py).

Same methods as the analysis scripts — occlusion (v14/v17) and SHAP (v18): an atom is
"removed" by zeroing its fingerprint embedding, and importance is the change in the
predicted probability of the positive class.
"""
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


@torch.no_grad()
def _prob(model, vectors, adj):
    for l in range(model.layer_hidden):
        vectors = F.normalize(model.update(adj, vectors, l), 2, 1)
    mol = model.attentions(vectors, adj).sum(0, keepdim=True)
    for l in range(model.layer_output):
        mol = torch.relu(model.W_output[l](mol))
    return torch.softmax(model.W_property(mol)[0], 0)[1].item()


def _inputs(predictor, smiles):
    _, fps, adj, _, _ = predictor.featurise([smiles])[smiles]
    adj = adj.float()
    with torch.no_grad():
        return predictor.model.embed_fingerprint(fps), adj


def occlusion(predictor, smiles):
    """Δp when each atom is removed on its own (+ = the atom supports the positive class)."""
    vecs, adj = _inputs(predictor, smiles)
    p0 = _prob(predictor.model, vecs, adj)
    out = []
    for i in range(len(vecs)):
        v = vecs.clone(); v[i] = 0
        out.append(p0 - _prob(predictor.model, v, adj))
    return p0, np.array(out)


def shap_values(predictor, smiles, nsamples="auto"):
    """KernelSHAP over atoms (background = all atoms removed); a few seconds per molecule."""
    import shap
    vecs, adj = _inputs(predictor, smiles)
    model = predictor.model

    def f(Z):
        z = torch.as_tensor(Z, dtype=vecs.dtype)
        return np.array([_prob(model, vecs * row[:, None], adj) for row in z])

    n = len(vecs)
    np.random.seed(0)
    explainer = shap.KernelExplainer(f, np.zeros((1, n)))
    sv = np.asarray(explainer.shap_values(np.ones((1, n)), nsamples=nsamples, silent=True)).reshape(-1)
    return _prob(model, vecs, adj), sv


def draw(smiles, values, legend="", size=(460, 340)):
    """PNG of the molecule, heavy atoms coloured red (+) / blue (−) by importance.
    preprocess adds explicit H after the heavy atoms, so the first n values are the heavy atoms."""
    mol = Chem.MolFromSmiles(smiles)
    imp = np.asarray(values)[:mol.GetNumAtoms()]
    scale = max(np.abs(imp).max(), 1e-9)
    colors = {i: ((1.0, float(1 - min(v / scale, 1)), float(1 - min(v / scale, 1))) if v > 0 else
                  (float(1 - min(-v / scale, 1)), float(1 - min(-v / scale, 1)), 1.0))
              for i, v in enumerate(imp)}
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(colors), highlightAtomColors=colors,
                                       highlightBonds=[], legend=legend)
    d.FinishDrawing()
    return d.GetDrawingText()


def top_atoms(smiles, values, k=3):
    mol = Chem.MolFromSmiles(smiles)
    imp = np.asarray(values)[:mol.GetNumAtoms()]
    return [(mol.GetAtomWithIdx(int(i)).GetSymbol(), int(i), float(imp[i])) for i in np.argsort(-np.abs(imp))[:k]]
