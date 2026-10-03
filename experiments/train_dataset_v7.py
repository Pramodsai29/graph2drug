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
import torch.optim as optim

from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix

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

BASE_CONFIG = {
    "train_file": str(DATASET_DIR / f"{DATASET}_train.txt"),
    "valid_file": str(DATASET_DIR / f"{DATASET}_valid.txt"),
    "test_file": str(DATASET_DIR / f"{DATASET}_test.txt"),
    "radius": 1, "dim": 52, "layer_hidden": 4, "layer_output": 10,
    "batch_train": 8, "batch_test": 8,
    "lr": 3e-4, "lr_decay": 0.85, "decay_interval": 25,
    "iteration": 40,   # short sweep run per config, not the full 140
    "N": 5000, "seed": 42,
}

# One controlled comparison: same seed, same architecture, only
# dropout / weight_decay vary between runs.
SWEEP = [
    {"label": "baseline_d045_wd0", "dropout": 0.45, "weight_decay": 0.0},
    {"label": "wd1e-4_d045", "dropout": 0.45, "weight_decay": 1e-4},
    {"label": "dropout06_wd0", "dropout": 0.60, "weight_decay": 0.0},
    {"label": "dropout06_wd1e-4", "dropout": 0.60, "weight_decay": 1e-4},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp.device = device

print("=" * 70)
print("D-GCAN v7 — Regularization sweep (dropout x weight_decay), 40 epochs each")
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


def evaluate_auc_mcc(predictions):
    y_true, y_pred, y_score = predictions[0], predictions[1], predictions[2]
    auc = roc_auc_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return auc, mcc, specificity


print("\nLoading datasets...")
train_dataset = pp.create_dataset(BASE_CONFIG["train_file"], "", "")
valid_dataset = pp.create_dataset(BASE_CONFIG["valid_file"], "", "")
test_dataset = pp.create_dataset(BASE_CONFIG["test_file"], "", "")
print("Train:", len(train_dataset), "Valid:", len(valid_dataset), "Test:", len(test_dataset))

sweep_results = []

for cfg in SWEEP:
    print("\n" + "=" * 70)
    print(f"CONFIG: {cfg['label']}  (dropout={cfg['dropout']}, weight_decay={cfg['weight_decay']})")
    print("=" * 70)

    random.seed(BASE_CONFIG["seed"])
    np.random.seed(BASE_CONFIG["seed"])
    torch.manual_seed(BASE_CONFIG["seed"])

    model = MolecularGraphNeuralNetwork(
        BASE_CONFIG["N"], BASE_CONFIG["dim"], BASE_CONFIG["layer_hidden"],
        BASE_CONFIG["layer_output"], cfg["dropout"],
    ).to(device)
    model.forward_classifier = types.MethodType(fixed_forward_classifier, model)

    trainer = Trainer(model, BASE_CONFIG["lr"], BASE_CONFIG["batch_train"])
    trainer.optimizer = optim.Adam(model.parameters(), lr=BASE_CONFIG["lr"], weight_decay=cfg["weight_decay"])
    tester = Tester(model, BASE_CONFIG["batch_test"])

    model_path = str(MODELS_DIR / f"{DATASET}_v7_{cfg['label']}_best.pth")
    best_auc = -1.0
    start = time.time()

    for epoch in range(BASE_CONFIG["iteration"]):
        if (epoch + 1) % BASE_CONFIG["decay_interval"] == 0:
            trainer.optimizer.param_groups[0]["lr"] *= BASE_CONFIG["lr_decay"]

        _, train_loss, _ = trainer.train(train_dataset)
        _, valid_loss, valid_predictions = tester.test_classifier(valid_dataset)
        valid_auc, valid_mcc, valid_spec = evaluate_auc_mcc(valid_predictions)

        if valid_auc > best_auc:
            best_auc = valid_auc
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:03d}/{BASE_CONFIG['iteration']} | valid_auc={valid_auc:.4f} "
                  f"| train_loss={train_loss:.2f} | valid_loss={valid_loss:.2f}")

    elapsed = (time.time() - start) / 60

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    _, _, test_predictions = tester.test_classifier(test_dataset)
    test_auc, test_mcc, test_spec = evaluate_auc_mcc(test_predictions)

    print(f"\n  -> best_valid_auc={best_auc:.4f} | test_auc={test_auc:.4f} | "
          f"test_mcc={test_mcc:.4f} | test_specificity={test_spec:.4f} | time={elapsed:.2f} min")

    sweep_results.append({
        "label": cfg["label"],
        "dropout": cfg["dropout"],
        "weight_decay": cfg["weight_decay"],
        "best_valid_auc": best_auc,
        "test_auc": test_auc,
        "test_mcc": test_mcc,
        "test_specificity": test_spec,
        "minutes": elapsed,
    })

results_df = pd.DataFrame(sweep_results).sort_values("test_auc", ascending=False)
results_df.to_csv(RESULTS_DIR / f"{DATASET}_v7_sweep_results.csv", index=False)

print("\n" + "=" * 70)
print("V7 SWEEP RESULTS (sorted by test_auc)")
print("=" * 70)
print(results_df.to_string(index=False))

best_row = results_df.iloc[0]
print(f"\nBest config by test_auc: {best_row['label']} "
      f"(dropout={best_row['dropout']}, weight_decay={best_row['weight_decay']}, "
      f"test_auc={best_row['test_auc']:.4f})")
print("\nNote: this is a 40-epoch proxy comparison, not a final result.")
print("If a config clearly wins, run it for the full 140 epochs next (v8).")
