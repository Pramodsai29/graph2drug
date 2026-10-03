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
}


def quiet(fn, *args):
    """preprocess.create_dataset prints every filename / failed SMILES — hide that."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def load_split(name, split):
    return quiet(pp.create_dataset, str(DATASET_DIR / f"{name}_{split}.txt"), "", "")


def build_vocabulary(name):
    """The fingerprint vocabulary is not stored in the checkpoint; preprocess builds it
    incrementally as files are read. Re-reading train, valid, test in the same order as
    training reproduces exactly the same fingerprint ids the model was trained on."""
    load_split(name, "train")
    load_split(name, "valid")
    return load_split(name, "test")


def featurise(smiles_list):
    """Turn new SMILES into model inputs. create_dataset skips its first line, so a
    header line is written first; the label column is a dummy (0)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("smiles\tlabel\n")
        for s in smiles_list:
            f.write(f"{s}\t0\n")
        path = f.name
    try:
        data = quiet(pp.create_dataset, path, "", "")
    finally:
        os.remove(path)
    return {d[0]: d for d in data}  # invalid SMILES are skipped by preprocess


def load_model(name):
    model = MolecularGraphNeuralNetwork(N, DIM, LAYER_HIDDEN, LAYER_OUTPUT, DROPOUT)
    state = torch.load(DATASETS[name]["checkpoint"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict(model, molecules, batch_size=8):
    """Probability of the positive class, computed the fixed way (raw logits -> softmax)."""
    probs = []
    for i in range(0, len(molecules), batch_size):
        batch = list(zip(*molecules[i:i + batch_size]))
        _, vectors = model.gnn(batch[:-1])
        for l in range(model.layer_output):
            vectors = torch.relu(model.W_output[l](vectors))
        logits = model.W_property(vectors)
        probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())
    return probs


def verify(name, model, test_set):
    probs = np.array(predict(model, test_set))
    labels = np.array([d[-1].item() for d in test_set])
    saved = pd.read_csv(DATASETS[name]["saved_predictions"]).iloc[:, 2].to_numpy()
    print(f"Test molecules        : {len(test_set)}")
    print(f"Test ROC-AUC (now)    : {roc_auc_score(labels, probs):.4f}")
    print(f"Test ROC-AUC (saved)  : {roc_auc_score(labels, saved):.4f}")
    print(f"Max |score difference|: {np.abs(probs - saved).max():.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", choices=DATASETS, default="BBBP")
    ap.add_argument("--smiles", nargs="+", help="one or more SMILES strings")
    ap.add_argument("--verify", action="store_true", help="re-score the test set")
    args = ap.parse_args()
    if not args.smiles and not args.verify:
        ap.error("give --smiles and/or --verify")

    cfg = DATASETS[args.dataset]
    print(f"Graph2Drug | D-GCAN (fixed) | dataset: {args.dataset} | "
          f"checkpoint: {cfg['checkpoint'].name}\n")
    test_set = build_vocabulary(args.dataset)
    model = load_model(args.dataset)

    if args.verify:
        verify(args.dataset, model, test_set)
        print()

    if args.smiles:
        feats = featurise(args.smiles)
        valid = [s for s in args.smiles if s in feats]
        probs = dict(zip(valid, predict(model, [feats[s] for s in valid])))
        for s in args.smiles:
            print(f"SMILES      : {s}")
            if s not in probs:
                print("Result      : invalid SMILES (RDKit could not parse it)\n")
                continue
            p = probs[s]
            verdict = cfg["positive"] if p > THRESHOLD else cfg["negative"]
            print(f"Probability : {p:.4f}")
            print(f"Prediction  : {verdict}\n")


if __name__ == "__main__":
    main()
