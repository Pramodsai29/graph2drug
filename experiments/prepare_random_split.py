"""
Random split counterpart to the scaffold splits (roadmap step 3).

For each dataset, pools the molecules of the existing scaffold split files
({DATASET}_train/valid/test.txt), shuffles them with seed 42 and re-cuts them to
exactly the same train/valid/test sizes. The molecule set, file format and sizes
are identical to the scaffold split — only which molecule lands in which split
changes — so any performance difference is attributable to the split strategy.

Each split file keeps a dummy first line, because preprocess.create_dataset always
discards the first line (the scaffold files lose their first molecule to the same
behaviour, so both strategies lose one molecule per file).

Output: datasets/processed/random/{DATASET}_{train,valid,test}.txt

Usage: python prepare_random_split.py [DATASET ...]   (default: all four)
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD_DIR = PROJECT_ROOT / "datasets" / "processed"
RANDOM_DIR = SCAFFOLD_DIR / "random"
RANDOM_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SPLITS = ("train", "valid", "test")
DATASETS = ("BBBP", "BACE", "ClinTox", "Tox21")


def read_lines(path):
    with open(path) as f:
        return [ln for ln in f.read().strip().split("\n") if ln.strip()]


def make_random_split(name):
    # line 0 of every scaffold file is never used by create_dataset; keep the
    # molecules that actually reach the model, and pad each new file the same way
    files = {s: read_lines(SCAFFOLD_DIR / f"{name}_{s}.txt") for s in SPLITS}
    used = {s: files[s][1:] for s in SPLITS}
    sizes = {s: len(used[s]) for s in SPLITS}
    pool = [ln for s in SPLITS for ln in used[s]]
    assert len(set(ln.split()[0] for ln in pool)) == len(pool), "duplicate molecules in pool"

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(pool))
    pool = [pool[i] for i in order]

    start, stats = 0, []
    for s in SPLITS:
        chunk = pool[start:start + sizes[s]]
        start += sizes[s]
        with open(RANDOM_DIR / f"{name}_{s}.txt", "w") as f:
            f.write("smiles\tlabel\n")  # discarded by create_dataset
            f.write("\n".join(chunk) + "\n")
        pos = sum(int(ln.split()[-1]) for ln in chunk)
        stats.append(f"{s}={len(chunk)} ({100 * pos / len(chunk):.1f}% pos)")
    print(f"{name:8s} " + "  ".join(stats))


if __name__ == "__main__":
    for name in sys.argv[1:] or DATASETS:
        make_random_split(name)
