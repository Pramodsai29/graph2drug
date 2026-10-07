"""
v11 — structural novelty of scaffold vs. random splits (analysis only, no training).

Generalises v6's Step C to all four datasets and both split strategies: for every
valid/test molecule, the maximum Tanimoto similarity (Morgan/ECFP4, 2048 bits) to
any training molecule. Lower similarity = the model is asked to generalise further
from what it trained on. Used to explain the scaffold-vs-random AUC gap (v10).

Only the molecules that actually reach the model are used: preprocess.create_dataset
discards each file's first line, so it is skipped here too.

Runs locally on CPU in about a minute.
Outputs: results/split_novelty_v11.csv, results/split_novelty_v11.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "datasets" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS = ("BBBP", "BACE", "ClinTox", "Tox21")
SPLIT_DIRS = {"scaffold": PROCESSED, "random": PROCESSED / "random"}

# same fingerprint as v6 (Morgan radius 2 = ECFP4, 2048 bits)
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def load(path):
    smiles = [ln.split()[0] for ln in path.read_text().strip().split("\n")[1:] if ln.strip()]
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fps.append(FPGEN.GetFingerprint(mol))
    return fps


def max_sim(query_fps, train_fps):
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in query_fps])


rows, dist = [], {}
for name in DATASETS:
    for split, d in SPLIT_DIRS.items():
        train = load(d / f"{name}_train.txt")
        for part in ("valid", "test"):
            sims = max_sim(load(d / f"{name}_{part}.txt"), train)
            dist[(name, split, part)] = sims
            rows.append({"dataset": name, "split": split, "part": part, "n": len(sims),
                         "mean_max_sim": sims.mean(), "median_max_sim": np.median(sims),
                         "frac_below_0.4": (sims < 0.4).mean()})
        print(f"{name:8s} {split:8s} test mean max-sim = {rows[-1]['mean_max_sim']:.4f}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "split_novelty_v11.csv", index=False)

# test-set distributions, scaffold vs random, one panel per dataset
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)
for ax, name in zip(axes, DATASETS):
    bins = np.linspace(0, 1, 26)
    ax.hist(dist[(name, "random", "test")], bins=bins, alpha=0.75, color="#2a78d6", label="random")
    ax.hist(dist[(name, "scaffold", "test")], bins=bins, alpha=0.75, color="#eb6834", label="scaffold")
    ax.set_title(name)
    ax.set_xlabel("Max Tanimoto similarity to train")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Test molecules")
axes[0].legend(frameon=False)
fig.suptitle("How similar is each test molecule to its nearest training molecule?")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "split_novelty_v11.png", dpi=150)

print()
print(df.pivot_table(index=["dataset", "part"], columns="split", values="mean_max_sim").round(4).to_string())
