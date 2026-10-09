"""
Graph2Drug — live prediction demo with a trained (fixed) D-GCAN checkpoint.

Examples:
    python experiments/demo_predict.py --dataset BBBP --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    python experiments/demo_predict.py --dataset BACE --smiles "<smiles>" "<smiles>"
    python experiments/demo_predict.py --dataset BBBP --verify     # re-score the test set

Runs on CPU in a few seconds. Requires the upstream repo at vendor/D-GCAN
(git clone https://github.com/JinYSun/D-GCAN.git vendor/D-GCAN).
"""
import argparse
import contextlib
import io
import os
import sys
import tempfile
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")  # invalid SMILES are reported by this script instead

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.append(str(PROJECT_ROOT / "vendor" / "D-GCAN" / "DGCAN"))

import preprocess as pp
from DGCAN import MolecularGraphNeuralNetwork

pp.device = torch.device("cpu")  # upstream hard-codes cuda

# Same architecture/hyperparameters the checkpoints were trained with (v5/v9).
N, DIM, LAYER_HIDDEN, LAYER_OUTPUT, DROPOUT = 5000, 52, 4, 10, 0.45
THRESHOLD = 0.15  # classification threshold used throughout the project's evaluation

DATASETS = {
    "BBBP": {
        "checkpoint": MODELS_DIR / "BBBP_v5_best_model.pth",
        "saved_predictions": RESULTS_DIR / "BBBP_v5_test_predictions.csv",
        "positive": "crosses the blood-brain barrier",
        "negative": "does not cross the blood-brain barrier",
    },
    "BACE": {
        "checkpoint": MODELS_DIR / "BACE_v9_seed42_best.pth",
        "saved_predictions": RESULTS_DIR / "BACE_v9_seed42_test_predictions.csv",
        "positive": "inhibits BACE-1 (Alzheimer's target)",
        "negative": "does not inhibit BACE-1",
    },
    # the D-GCAN paper's own task (FDA drugs vs ZINC), v16 full model, random split
    "druglikeRandom": {
        "checkpoint": MODELS_DIR / "druglikeRandom_v16_full_seed42_best.pth",
        "saved_predictions": RESULTS_DIR / "druglikeRandom_v16_full_seed42_test_predictions.csv",
        "positive": "drug-like",
        "negative": "not drug-like",
    },
}

VOCAB_NAMES = ("atom_dict", "bond_dict", "fingerprint_dict", "edge_dict")
# preprocess's vocabulary is module-global state; the web app serves several sessions
# on different threads, so every swap-in/use/swap-out must happen under one lock.
_VOCAB_LOCK = threading.RLock()


def quiet(fn, *args):
    """preprocess.create_dataset prints every filename / failed SMILES — hide that."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


class Predictor:
    """A trained checkpoint plus the fingerprint vocabulary it was trained with.

    preprocess keeps its vocabulary in module-level dicts that grow as files are read,
    and the checkpoint does not store it. Re-reading train, valid, test in the same
    order as training rebuilds exactly the same fingerprint ids. Each Predictor owns its
    own dicts and swaps them into preprocess while it works, so BBBP and BACE predictors
    can live in the same process without mixing vocabularies."""

    def __init__(self, name):
        self.name = name
        self.cfg = DATASETS[name]
        self.vocab = tuple(defaultdict(self._counter(n)) for n in VOCAB_NAMES)
        with self._active():
            quiet(pp.create_dataset, str(DATASET_DIR / f"{name}_train.txt"), "", "")
            quiet(pp.create_dataset, str(DATASET_DIR / f"{name}_valid.txt"), "", "")
            self.test_set = quiet(pp.create_dataset, str(DATASET_DIR / f"{name}_test.txt"), "", "")
        self.model = MolecularGraphNeuralNetwork(N, DIM, LAYER_HIDDEN, LAYER_OUTPUT, DROPOUT)
        self.model.load_state_dict(torch.load(self.cfg["checkpoint"], map_location="cpu"))
        self.model.eval()

    @staticmethod
    def _counter(attr):
        # same behaviour as preprocess's own factories: next unused id
        return lambda: len(getattr(pp, attr))

    @contextlib.contextmanager
    def _active(self):
        with _VOCAB_LOCK:
            saved = tuple(getattr(pp, n) for n in VOCAB_NAMES)
            for n, d in zip(VOCAB_NAMES, self.vocab):
                setattr(pp, n, d)
            try:
                yield
            finally:
                for n, d in zip(VOCAB_NAMES, saved):
                    setattr(pp, n, d)

    def featurise(self, smiles_list):
        """New SMILES -> model inputs. create_dataset skips its first line, so a header
        is written first; the label column is a dummy (0). Invalid SMILES are dropped."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("smiles\tlabel\n")
            for s in smiles_list:
                f.write(f"{s}\t0\n")
            path = f.name
        try:
            with self._active():
                data = quiet(pp.create_dataset, path, "", "")
        finally:
            os.remove(path)
        return {d[0]: d for d in data}

    @torch.no_grad()
    def _scores(self, molecules, batch_size=8):
        """Probability of the positive class, computed the fixed way (raw logits -> softmax)."""
        probs = []
        for i in range(0, len(molecules), batch_size):
            batch = list(zip(*molecules[i:i + batch_size]))
            _, vectors = self.model.gnn(batch[:-1])
            for l in range(self.model.layer_output):
                vectors = torch.relu(self.model.W_output[l](vectors))
            logits = self.model.W_property(vectors)
            probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())
        return probs

    def predict(self, smiles_list):
        """{smiles: probability or None if invalid}"""
        feats = self.featurise(smiles_list)
        valid = [s for s in smiles_list if s in feats]
        probs = dict(zip(valid, self._scores([feats[s] for s in valid])))
        return {s: probs.get(s) for s in smiles_list}

    def verdict(self, p):
        return self.cfg["positive"] if p > THRESHOLD else self.cfg["negative"]

    def verify(self):
        probs = np.array(self._scores(self.test_set))
        labels = np.array([d[-1].item() for d in self.test_set])
        saved = pd.read_csv(self.cfg["saved_predictions"]).iloc[:, 2].to_numpy()
        return {"n": len(self.test_set),
                "auc_now": roc_auc_score(labels, probs),
                "auc_saved": roc_auc_score(labels, saved),
                "max_diff": float(np.abs(probs - saved).max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", choices=DATASETS, default="BBBP")
    ap.add_argument("--smiles", nargs="+", help="one or more SMILES strings")
    ap.add_argument("--verify", action="store_true", help="re-score the test set")
    args = ap.parse_args()
    if not args.smiles and not args.verify:
        ap.error("give --smiles and/or --verify")

    pred = Predictor(args.dataset)
    print(f"Graph2Drug | D-GCAN (fixed) | dataset: {args.dataset} | "
          f"checkpoint: {pred.cfg['checkpoint'].name}\n")

    if args.verify:
        v = pred.verify()
        print(f"Test molecules        : {v['n']}")
        print(f"Test ROC-AUC (now)    : {v['auc_now']:.4f}")
        print(f"Test ROC-AUC (saved)  : {v['auc_saved']:.4f}")
        print(f"Max |score difference|: {v['max_diff']:.2e}\n")

    if args.smiles:
        for s, p in pred.predict(args.smiles).items():
            print(f"SMILES      : {s}")
            if p is None:
                print("Result      : invalid SMILES (RDKit could not parse it)\n")
                continue
            print(f"Probability : {p:.4f}")
            print(f"Prediction  : {pred.verdict(p)}\n")


if __name__ == "__main__":
    main()
