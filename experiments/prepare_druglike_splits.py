"""
Random and scaffold 80/10/10 splits of the D-GCAN authors' own drug-likeness data
(v16 — re-testing the paper's ablation claims on its own dataset).

Pools the molecules of vendor/D-GCAN/dataset/data_train.txt and data_test.txt that
preprocess.create_dataset actually reads (each file's first line is skipped), drops
RDKit-invalid SMILES and duplicate structures, then writes two splits:
  - random   : shuffled with seed 42 (the paper used a random 8:1:1 split)
  - scaffold : Bemis–Murcko scaffold split, BALANCED variant (Chemprop's
               "scaffold_balanced"): scaffold groups larger than half the test size go
               to train first, the rest are shuffled (seed 42) and filled into train,
               valid, test. No scaffold appears in two splits.
               The DeepChem largest-first variant used for BBBP/BACE/ClinTox/Tox21 is
               degenerate here: non-drugs (ZINC) share a few large scaffolds that all
               land in train, so valid/test end up 100% drugs (reported, not used).
Each output file starts with a dummy header line, because create_dataset discards
the first line.

Output: datasets/processed/druglike{Random,Scaffold}_{train,valid,test}.txt
Usage:  python experiments/prepare_druglike_splits.py
"""
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "vendor" / "D-GCAN" / "dataset"
OUT = ROOT / "datasets" / "processed"
SEED = 42

pool, seen = [], set()
for name in ("data_train.txt", "data_test.txt"):
    for ln in (SRC / name).read_text().strip().split("\n")[1:]:
        parts = ln.split()
        if len(parts) != 2 or parts[1] not in ("0", "1"):
            continue
        mol = Chem.MolFromSmiles(parts[0])
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            continue
        seen.add(canon)
        pool.append((parts[0], int(parts[1]), mol))
n = len(pool)
print(f"pooled {n} unique valid molecules ({sum(p[1] for p in pool)} drugs)")

n_train, n_valid = int(0.8 * n), int(0.9 * n) - int(0.8 * n)

# random split
order = np.random.RandomState(SEED).permutation(n)
random_idx = {"train": order[:n_train], "valid": order[n_train:n_train + n_valid], "test": order[n_train + n_valid:]}

# scaffold split, balanced variant (Chemprop scaffold_balanced)
groups = {}
for i, (_, _, mol) in enumerate(pool):
    groups.setdefault(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False), []).append(i)
n_test = n - n_train - n_valid
big = [g for g in groups.values() if len(g) > n_test / 2]
small = [g for g in groups.values() if len(g) <= n_test / 2]
rng = np.random.RandomState(SEED)
rng.shuffle(big)
rng.shuffle(small)
train, valid, test = [], [], []
for g in big + small:
    if len(train) + len(g) <= n_train:
        train += g
    elif len(valid) + len(g) <= n_valid:
        valid += g
    else:
        test += g
scaffold_idx = {"train": train, "valid": valid, "test": test}

for tag, idx in (("Random", random_idx), ("Scaffold", scaffold_idx)):
    stats = []
    for part, ids in idx.items():
        rows = [pool[i] for i in ids]
        with open(OUT / f"druglike{tag}_{part}.txt", "w") as f:
            f.write("smiles\tlabel\n")
            f.write("\n".join(f"{s}\t{y}" for s, y, _ in rows) + "\n")
        stats.append(f"{part}={len(rows)} ({100 * sum(r[1] for r in rows) / len(rows):.1f}% drugs)")
    print(f"{tag:8s} " + "  ".join(stats))
