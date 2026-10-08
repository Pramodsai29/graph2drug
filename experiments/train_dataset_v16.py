"""
v16 — the D-GCAN paper's own ablation, re-run on the paper's own drug-likeness data
(roadmap follow-up to step 5). The paper (Sun et al. 2022, single random 8:1:1 split,
one run) reports: full D-GCAN ACC 0.923; "GAT" (no graph convolution) 0.862; "GCNN"
(no attention) 0.862 — i.e. attention +4.0%, convolution +6.1%.

Here: identical machinery to v13 (logits fix, original hyperparameters, best checkpoint
by validation AUC), on the authors' data re-split 80/10/10 by
experiments/prepare_druglike_splits.py:
  DATASET = druglikeRandom    random split (as in the paper, but with a validation set)
  DATASET = druglikeScaffold  balanced scaffold split
Variants: full (paper's D-GCAN), no_gcn (paper's "GAT"), no_gat (paper's "GCNN").
Seeds 42-44 by default (GPU budget). Reports ROC-AUC from scores AND accuracy at the
paper's 0.15 threshold.

Colab: set DATASET_OVERRIDE, ABLATIONS_OVERRIDE, SEEDS_OVERRIDE in the kernel before
`colab exec -f`; locally use env vars DATASET / ABLATIONS / SEEDS / TAG_SUFFIX.
"""
import importlib
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
from torch import nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix,
)

ON_COLAB = Path("/content/datasets").exists()
ROOT = Path("/content") if ON_COLAB else Path(__file__).resolve().parents[1]
DGCAN_REPO = ROOT / "D-GCAN" if ON_COLAB else ROOT / "vendor" / "D-GCAN"
DATASET_DIR = ROOT / "datasets" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
if not (DGCAN_REPO / "DGCAN" / "DGCAN.py").exists():
    subprocess.run(["git", "clone", "https://github.com/JinYSun/D-GCAN.git", str(DGCAN_REPO)], check=True)
sys.path.append(str(DGCAN_REPO / "DGCAN"))

import preprocess as pp
import DGCAN as dg

DATASET = globals().get("DATASET_OVERRIDE", os.environ.get("DATASET", "BBBP"))
ABLATIONS = globals().get("ABLATIONS_OVERRIDE",
                          os.environ.get("ABLATIONS", "full,no_gcn,no_gat").split(","))

CONFIG = {"radius": 1, "dim": 52, "layer_hidden": 4, "layer_output": 10, "dropout": 0.45,
          "batch_train": 8, "batch_test": 8, "lr": 3e-4, "lr_decay": 0.85, "decay_interval": 25,
          "iteration": int(os.environ.get("EPOCHS", 140)), "N": 5000}  # EPOCHS env: smoke tests only
SEEDS = [int(x) for x in globals().get("SEEDS_OVERRIDE", os.environ.get("SEEDS", "42,43,44").split(","))]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixed_forward_classifier(self, data_batch, train):
    """The v4/v5 logits fix (identical to v8/v9)."""
    inputs = data_batch[:-1]
    correct_labels = torch.cat(data_batch[-1])
    with torch.set_grad_enabled(train):
        Smiles, vectors = self.gnn(inputs)
        for l in range(self.layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        logits = self.W_property(vectors)
        loss = F.cross_entropy(logits, correct_labels)
        probs = F.softmax(logits, dim=1)
    predicted_scores = [s[1] for s in probs.detach().to("cpu").numpy()]
    return Smiles, loss, predicted_scores, correct_labels.to("cpu").numpy()


# ---------------------------------------------------------------- ablation patches
def gnn_no_gat(self, inputs):
    """gnn() with the GAT block replaced by a per-atom projection (no neighbour mixing)."""
    Smiles, fingerprints, adjacencies, molecular_sizes = inputs
    fingerprints = torch.cat(fingerprints)
    adj = self.pad(adjacencies, 0)
    vectors = self.embed_fingerprint(fingerprints)
    for l in range(self.layer_hidden):
        vectors = F.normalize(self.update(adj, vectors, l), 2, 1)
    atom_out = F.softmax(F.elu(self.no_gat_proj(vectors)), dim=1)  # same output form as GAT
    return Smiles, self.sum(atom_out, molecular_sizes)


def uniform_attention_forward(self, input, adj):
    """GraphAttentionLayer.forward with learned coefficients replaced by a uniform
    average over neighbours (isolated atoms keep their own features)."""
    h = torch.mm(input, self.W)
    mask = (adj > 0).float()
    deg = mask.sum(1, keepdim=True)
    attention = torch.where(deg > 0, mask / deg.clamp(min=1), torch.eye(len(adj), device=adj.device))
    attention = F.dropout(attention, self.dropout, training=self.training)
    h_prime = torch.matmul(attention, h)
    return F.elu(h_prime) if self.concat else h_prime


def build_model(ablation):
    layer_hidden = 0 if ablation == "no_gcn" else CONFIG["layer_hidden"]
    model = dg.MolecularGraphNeuralNetwork(CONFIG["N"], CONFIG["dim"], layer_hidden,
                                           CONFIG["layer_output"], CONFIG["dropout"]).to(device)
    model.forward_classifier = types.MethodType(fixed_forward_classifier, model)
    if ablation == "no_gat":
        model.no_gat_proj = nn.Linear(CONFIG["dim"], 56).to(device)  # registered before the optimizer
        model.gnn = types.MethodType(gnn_no_gat, model)
    if ablation == "uniform_attn":
        for layer in model.attentions.modules():
            if isinstance(layer, dg.GraphAttentionLayer):
                layer.forward = types.MethodType(uniform_attention_forward, layer)
    return model


def evaluate(predictions):
    y_true, y_pred, y_score = predictions[0], predictions[1], predictions[2]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"auc": roc_auc_score(y_true, y_score), "pr_auc": average_precision_score(y_true, y_score),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred), "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0), "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred), "specificity": tn / (tn + fp) if (tn + fp) else 0.0}


print("=" * 70)
print(f"D-GCAN v16 paper-ablation replication — {DATASET} | {', '.join(ABLATIONS)} | device {device}")
print("=" * 70)

for ablation in ABLATIONS:
    # fresh preprocess state per ablation: vocab dicts start empty, radius per variant
    pp = importlib.reload(pp)
    pp.device = device
    dg.device = device
    pp.radius = 0 if ablation == "no_fp" else CONFIG["radius"]
    train_dataset = pp.create_dataset(str(DATASET_DIR / f"{DATASET}_train.txt"), "", "")
    valid_dataset = pp.create_dataset(str(DATASET_DIR / f"{DATASET}_valid.txt"), "", "")
    test_dataset = pp.create_dataset(str(DATASET_DIR / f"{DATASET}_test.txt"), "", "")
    print(f"\n[{ablation}] radius={pp.radius} | fingerprint vocab={len(pp.fingerprint_dict)} | "
          f"train/valid/test = {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}")

    tag = f"v16_{ablation}{os.environ.get('TAG_SUFFIX', '')}"  # e.g. "_cpu" for laptop runs
    rows, t_abl = [], time.time()
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = build_model(ablation)
        trainer = dg.Trainer(model, CONFIG["lr"], CONFIG["batch_train"])
        tester = dg.Tester(model, CONFIG["batch_test"])
        path = str(MODELS_DIR / f"{DATASET}_{tag}_seed{seed}_best.pth")
        best_auc, best_epoch, log, t0 = -1.0, 0, [], time.time()
        for epoch in range(CONFIG["iteration"]):
            if (epoch + 1) % CONFIG["decay_interval"] == 0:
                trainer.optimizer.param_groups[0]["lr"] *= CONFIG["lr_decay"]
            # like v8/v9, the model is never switched to eval mode during training,
            # so per-epoch validation runs with dropout on (kept for comparability)
            train_auc, train_loss, _ = trainer.train(train_dataset)
            _, _, vp = tester.test_classifier(valid_dataset)
            v = evaluate(vp)
            log.append({"epoch": epoch + 1, "train_loss": train_loss, "train_auc": train_auc,
                        "valid_auc": v["auc"], "valid_pr_auc": v["pr_auc"]})
            if v["auc"] > best_auc:
                best_auc, best_epoch = v["auc"], epoch + 1
                torch.save(model.state_dict(), path)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        _, _, tp = tester.test_classifier(test_dataset)
        m = evaluate(tp)
        pd.DataFrame(log).to_csv(RESULTS_DIR / f"{DATASET}_{tag}_seed{seed}_training_log.csv", index=False)
        pd.DataFrame({"y_true": tp[0], "y_pred": tp[1], "y_score": tp[2]}).to_csv(
            RESULTS_DIR / f"{DATASET}_{tag}_seed{seed}_test_predictions.csv", index=False)
        minutes = (time.time() - t0) / 60
        print(f"  -> {ablation} seed {seed}: best_valid_auc={best_auc:.4f} (epoch {best_epoch}) | "
              f"test_auc={m['auc']:.4f} | test_mcc={m['mcc']:.4f} | time={minutes:.2f} min")
        rows.append({"seed": seed, "best_valid_auc": best_auc, "best_epoch": best_epoch,
                     **{f"test_{k}": val for k, val in m.items()}, "minutes": minutes})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / f"{DATASET}_{tag}_multiseed_results.csv", index=False)
    summary = {"Dataset": DATASET, "Ablation": ablation, "N_seeds": len(SEEDS),
               "Total_minutes": (time.time() - t_abl) / 60}
    for k in ["auc", "pr_auc", "balanced_accuracy", "accuracy", "precision", "recall", "f1", "mcc", "specificity"]:
        summary[f"test_{k}_mean"] = res[f"test_{k}"].mean()
        summary[f"test_{k}_std"] = res[f"test_{k}"].std()
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{DATASET}_{tag}_summary.csv", index=False)
    print(f"{ablation} test_auc        : {summary['test_auc_mean']:.4f} +/- {summary['test_auc_std']:.4f}")
