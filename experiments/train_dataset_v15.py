"""
v15 — reproduce D-GCAN on the AUTHORS' OWN drug-likeness data and protocol, with the
original (buggy) code and with our logits fix.

Question: does the sigmoid -> cross_entropy bug (found on BBBP, v1-v5) also break
training on the data the paper reported (ACC 0.923, "AUC" 0.951)?

Replicates the authors' DGCAN/train.py + run.py exactly:
  - data: dataset/data_train.txt (train) and dataset/data_test.txt (test) from the
    upstream repo; preprocess.create_dataset (drops each file's first line, as theirs does)
  - no validation set: the test set is evaluated every epoch and the LAST-epoch model is
    reported (their script saves the model every epoch, then evaluates it once more)
  - hyperparameters from run.py (radius 1, dim 52, 4 GCN layers, 10 dense, dropout 0.45,
    batch 8, lr 3e-4, x0.85 every 25 epochs, 140 epochs, N 5000)
  - np.random.seed(seed) -> shuffle train -> torch.manual_seed(seed) -> np.random.seed(seed)
    (their seed is 0)
Metrics reported:
  - acc                  threshold 0.15 (theirs)
  - auc_hardlabel        what train.py calls "auc": roc_curve(true, 0/1 PREDICTIONS)
  - auc_score            proper ROC-AUC from the predicted probability
  - score_std            spread of predicted scores (collapse check, as in v2)

MODE "bug" = upstream forward_classifier untouched; "fix" = v4/v5 logits fix.
Colab: set MODES_OVERRIDE / SEEDS_OVERRIDE in the kernel before `colab exec -f`.
Outputs: results/druglike_v15_{mode}_seed{s}_log.csv, results/druglike_v15_runs.csv
"""
import os
import random
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve

ON_COLAB = Path("/content").exists() and Path("/content/results").exists()
ROOT = Path("/content") if ON_COLAB else Path(__file__).resolve().parents[1]
DGCAN_REPO = ROOT / "D-GCAN" if ON_COLAB else ROOT / "vendor" / "D-GCAN"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if not (DGCAN_REPO / "DGCAN" / "DGCAN.py").exists():
    subprocess.run(["git", "clone", "https://github.com/JinYSun/D-GCAN.git", str(DGCAN_REPO)], check=True)
sys.path.append(str(DGCAN_REPO / "DGCAN"))

import preprocess as pp
import DGCAN as dg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp.device = device
dg.device = device

MODES = globals().get("MODES_OVERRIDE", os.environ.get("MODES", "bug,fix").split(","))
SEEDS = [int(s) for s in globals().get("SEEDS_OVERRIDE", os.environ.get("SEEDS", "0,42,43").split(","))]
EPOCHS = int(os.environ.get("EPOCHS", 140))  # EPOCHS env: smoke tests only
CFG = dict(N=5000, dim=52, layer_hidden=4, layer_output=10, dropout=0.45,
           batch=8, lr=3e-4, lr_decay=0.85, decay_interval=25)


def fixed_forward_classifier(self, data_batch, train):
    """v4/v5 logits fix (identical to v8/v9)."""
    inputs = data_batch[:-1]
    correct_labels = torch.cat(data_batch[-1])
    with torch.set_grad_enabled(train):
        Smiles, vectors = self.gnn(inputs)
        for l in range(self.layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        logits = self.W_property(vectors)
        loss = F.cross_entropy(logits, correct_labels)
        probs = F.softmax(logits, dim=1)
    return Smiles, loss, [s[1] for s in probs.detach().cpu().numpy()], correct_labels.cpu().numpy()


def metrics(pred):
    y, yhat, score = pred[0], pred[1], pred[2]
    fpr, tpr, _ = roc_curve(y, yhat)  # exactly as the authors' train.py
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"acc": (tp + tn) / len(y), "auc_hardlabel": auc(fpr, tpr), "auc_score": roc_auc_score(y, score),
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0, "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "score_std": float(np.std(score))}


data_dir = DGCAN_REPO / "dataset"
train_raw = pp.create_dataset(str(data_dir / "data_train.txt"), "", "")
test_set = pp.create_dataset(str(data_dir / "data_test.txt"), "", "")
print(f"D-GCAN v15 — authors' drug-likeness data | modes {MODES} | seeds {SEEDS} | {EPOCHS} epochs | {device}")
print(f"train {len(train_raw)} | test {len(test_set)} | fingerprint vocab {len(pp.fingerprint_dict)}")

rows = []
for mode in MODES:
    for seed in SEEDS:
        train_set = list(train_raw)
        np.random.seed(seed)
        np.random.shuffle(train_set)
        torch.manual_seed(seed)
        random.seed(seed)
        model = dg.MolecularGraphNeuralNetwork(CFG["N"], CFG["dim"], CFG["layer_hidden"],
                                               CFG["layer_output"], CFG["dropout"]).to(device)
        if mode == "fix":
            model.forward_classifier = types.MethodType(fixed_forward_classifier, model)
        trainer = dg.Trainer(model, CFG["lr"], CFG["batch"])
        tester = dg.Tester(model, CFG["batch"])
        np.random.seed(seed)
        log, t0 = [], time.time()
        for epoch in range(1, EPOCHS + 1):
            if epoch % CFG["decay_interval"] == 0:
                trainer.optimizer.param_groups[0]["lr"] *= CFG["lr_decay"]
            train_acc_auc, train_loss, _ = trainer.train(train_set)
            _, test_loss, tp = tester.test_classifier(test_set)
            m = metrics(tp)
            log.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, **m})
        model.eval()
        _, _, tp = tester.test_classifier(test_set)
        final = metrics(tp)
        minutes = (time.time() - t0) / 60
        pd.DataFrame(log).to_csv(RESULTS_DIR / f"druglike_v15_{mode}_seed{seed}_log.csv", index=False)
        print(f"  -> {mode} seed {seed}: acc={final['acc']:.4f} | auc_hardlabel={final['auc_hardlabel']:.4f} | "
              f"auc_score={final['auc_score']:.4f} | score_std={final['score_std']:.4f} | {minutes:.1f} min", flush=True)
        rows.append({"mode": mode, "seed": seed, **final, "minutes": minutes})
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "druglike_v15_runs.csv", index=False)

df = pd.DataFrame(rows)
print("\n" + df.groupby("mode")[["acc", "auc_hardlabel", "auc_score", "score_std"]].agg(["mean", "std"]).round(4).to_string())
print("V15 DONE")
