import sys
import time
import types
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
)

# ==========================================================
# Colab paths (this script runs on a Colab VM, /content working dir)
# ==========================================================

CONTENT = Path("/content")
DGCAN_REPO = CONTENT / "D-GCAN"
DATASET_DIR = CONTENT / "datasets" / "processed"
MODELS_DIR = CONTENT / "models"
RESULTS_DIR = CONTENT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not (DGCAN_REPO / "DGCAN" / "DGCAN.py").exists():
    print("Cloning D-GCAN repo...")
    subprocess.run(
        ["git", "clone", "https://github.com/JinYSun/D-GCAN.git", str(DGCAN_REPO)],
        check=True,
    )

sys.path.append(str(DGCAN_REPO / "DGCAN"))

import preprocess as pp
from DGCAN import MolecularGraphNeuralNetwork, Trainer, Tester

DATASET = "BBBP"

# Same hyperparameters as the original config (v1/v5) — the v7 sweep found no
# robust reason to change dropout or add weight decay. Only the sigmoid-before-
# cross_entropy fix (v4/v5) is applied. This run's purpose is statistically
# sound reporting (mean +/- std across seeds), not further tuning.
CONFIG = {
    "train_file": str(DATASET_DIR / f"{DATASET}_train.txt"),
    "valid_file": str(DATASET_DIR / f"{DATASET}_valid.txt"),
    "test_file": str(DATASET_DIR / f"{DATASET}_test.txt"),
    "radius": 1, "dim": 52, "layer_hidden": 4, "layer_output": 10,
    "dropout": 0.45,
    "batch_train": 8, "batch_test": 8,
    "lr": 3e-4, "lr_decay": 0.85, "decay_interval": 25,
    "iteration": 140,
    "N": 5000,
}

SEEDS = [42, 43, 44, 45, 46]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp.device = device

print("=" * 70)
print("D-GCAN v8 — Multi-seed BBBP baseline (5 seeds x 140 epochs)")
print("=" * 70)
print("Device:", device)


def fixed_forward_classifier(self, data_batch, train):
    inputs = data_batch[:-1]
    correct_labels = torch.cat(data_batch[-1])
    if train:
        Smiles, molecular_vectors = self.gnn(inputs)
        vectors = molecular_vectors
        for l in range(self.layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        logits = self.W_property(vectors)
        loss = F.cross_entropy(logits, correct_labels)
        probs = F.softmax(logits, dim=1)
        predicted_scores = probs.to("cpu").data.numpy()
        predicted_scores = [s[1] for s in predicted_scores]
        correct_labels = correct_labels.to("cpu").data.numpy()
        return Smiles, loss, predicted_scores, correct_labels
    else:
        with torch.no_grad():
            Smiles, molecular_vectors = self.gnn(inputs)
            vectors = molecular_vectors
            for l in range(self.layer_output):
                vectors = torch.relu(self.W_output[l](vectors))
            logits = self.W_property(vectors)
            loss = F.cross_entropy(logits, correct_labels)
            probs = F.softmax(logits, dim=1)
        predicted_scores = probs.to("cpu").data.numpy()
        predicted_scores = [s[1] for s in predicted_scores]
        correct_labels = correct_labels.to("cpu").data.numpy()
        return Smiles, loss, predicted_scores, correct_labels


def evaluate(predictions):
    y_true, y_pred, y_score = predictions[0], predictions[1], predictions[2]
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"auc": auc, "accuracy": acc, "precision": precision, "recall": recall,
            "f1": f1, "mcc": mcc, "specificity": specificity}


print("\nLoading datasets...")
train_dataset = pp.create_dataset(CONFIG["train_file"], "", "")
valid_dataset = pp.create_dataset(CONFIG["valid_file"], "", "")
test_dataset = pp.create_dataset(CONFIG["test_file"], "", "")
print("Train:", len(train_dataset), "Valid:", len(valid_dataset), "Test:", len(test_dataset))

seed_results = []
overall_start = time.time()

for seed in SEEDS:
    print("\n" + "=" * 70)
    print(f"SEED {seed}")
    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = MolecularGraphNeuralNetwork(
        CONFIG["N"], CONFIG["dim"], CONFIG["layer_hidden"], CONFIG["layer_output"], CONFIG["dropout"]
    ).to(device)
    model.forward_classifier = types.MethodType(fixed_forward_classifier, model)

    trainer = Trainer(model, CONFIG["lr"], CONFIG["batch_train"])
    tester = Tester(model, CONFIG["batch_test"])

    model_path = str(MODELS_DIR / f"{DATASET}_v8_seed{seed}_best.pth")
    best_auc = -1.0
    start = time.time()

    for epoch in range(CONFIG["iteration"]):
        if (epoch + 1) % CONFIG["decay_interval"] == 0:
            trainer.optimizer.param_groups[0]["lr"] *= CONFIG["lr_decay"]

        trainer.train(train_dataset)
        _, _, valid_predictions = tester.test_classifier(valid_dataset)
        valid_metrics = evaluate(valid_predictions)

        if valid_metrics["auc"] > best_auc:
            best_auc = valid_metrics["auc"]
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 35 == 0:
            print(f"  epoch {epoch+1:03d}/{CONFIG['iteration']} | valid_auc={valid_metrics['auc']:.4f}")

    elapsed = (time.time() - start) / 60

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    _, _, test_predictions = tester.test_classifier(test_dataset)
    test_metrics = evaluate(test_predictions)

    print(f"\n  -> seed {seed}: best_valid_auc={best_auc:.4f} | test_auc={test_metrics['auc']:.4f} | "
          f"test_mcc={test_metrics['mcc']:.4f} | test_specificity={test_metrics['specificity']:.4f} | "
          f"time={elapsed:.2f} min")

    seed_results.append({
        "seed": seed,
        "best_valid_auc": best_auc,
        "test_auc": test_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_mcc": test_metrics["mcc"],
        "test_specificity": test_metrics["specificity"],
        "minutes": elapsed,
    })

results_df = pd.DataFrame(seed_results)
results_df.to_csv(RESULTS_DIR / f"{DATASET}_v8_multiseed_results.csv", index=False)

total_time = (time.time() - overall_start) / 60

print("\n" + "=" * 70)
print("V8 MULTI-SEED RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))

summary_metrics = ["test_auc", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_mcc", "test_specificity"]
summary = {"Dataset": DATASET, "N_seeds": len(SEEDS), "Total_minutes": total_time}
for m in summary_metrics:
    summary[f"{m}_mean"] = results_df[m].mean()
    summary[f"{m}_std"] = results_df[m].std()

pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{DATASET}_v8_summary.csv", index=False)

print("\n" + "-" * 70)
print("MEAN +/- STD ACROSS SEEDS (report this, not a single-seed number)")
print("-" * 70)
for m in summary_metrics:
    print(f"{m:<16}: {summary[f'{m}_mean']:.4f} +/- {summary[f'{m}_std']:.4f}")
print(f"\nTotal time for {len(SEEDS)} seeds: {total_time:.2f} minutes")
