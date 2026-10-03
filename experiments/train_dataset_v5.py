import os
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
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
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

CONFIG = {
    "dataset": DATASET,
    "train_file": str(DATASET_DIR / f"{DATASET}_train.txt"),
    "valid_file": str(DATASET_DIR / f"{DATASET}_valid.txt"),
    "test_file": str(DATASET_DIR / f"{DATASET}_test.txt"),
    "radius": 1,
    "dim": 52,
    "layer_hidden": 4,
    "layer_output": 10,
    "dropout": 0.45,
    "batch_train": 8,
    "batch_test": 8,
    "lr": 3e-4,
    "lr_decay": 0.85,
    "decay_interval": 25,
    "iteration": 140,   # full run — v4 smoke test (10 epochs) already validated the fix
    "N": 5000,
    "threshold": 0.15,
    "seed": 42,
    "model_path": str(MODELS_DIR / f"{DATASET}_v5_best_model.pth"),
    "log_path": str(RESULTS_DIR / f"{DATASET}_v5_training_log.csv"),
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp.device = device

print("=" * 70)
print("D-GCAN v5 — FULL 140-EPOCH RUN with the sigmoid-before-CE fix")
print("=" * 70)
print("Dataset:", CONFIG["dataset"])
print("Device :", device)

# ==========================================================
# Same fix as v4: raw logits -> cross_entropy, softmax for reporting only.
# Monkey-patched onto the model instance; vendor/D-GCAN/DGCAN.py stays untouched.
# ==========================================================

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
    return {
        "auc": auc, "accuracy": acc, "precision": precision, "recall": recall,
        "f1": f1, "mcc": mcc, "specificity": specificity,
    }


print("\nLoading datasets...\n")
train_dataset = pp.create_dataset(CONFIG["train_file"], "", "")
valid_dataset = pp.create_dataset(CONFIG["valid_file"], "", "")
test_dataset = pp.create_dataset(CONFIG["test_file"], "", "")
print("Train :", len(train_dataset))
print("Valid :", len(valid_dataset))
print("Test  :", len(test_dataset))

print("\nCreating Model (fresh init, fix applied)...\n")
model = MolecularGraphNeuralNetwork(
    CONFIG["N"], CONFIG["dim"], CONFIG["layer_hidden"], CONFIG["layer_output"], CONFIG["dropout"]
).to(device)
model.forward_classifier = types.MethodType(fixed_forward_classifier, model)

trainer = Trainer(model, CONFIG["lr"], CONFIG["batch_train"])
tester = Tester(model, CONFIG["batch_test"])

print("Model Ready")
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("=" * 70)

history = []
best_auc = -1.0

print("\nStarting Training (140 epochs)...\n")
training_start = time.time()

for epoch in range(CONFIG["iteration"]):
    if (epoch + 1) % CONFIG["decay_interval"] == 0:
        trainer.optimizer.param_groups[0]["lr"] *= CONFIG["lr_decay"]
    current_lr = trainer.optimizer.param_groups[0]["lr"]

    epoch_start = time.time()

    train_auc, train_loss, train_predictions = trainer.train(train_dataset)
    train_metrics = evaluate(train_predictions)

    _, valid_loss, valid_predictions = tester.test_classifier(valid_dataset)
    valid_metrics = evaluate(valid_predictions)

    epoch_time = time.time() - epoch_start

    if valid_metrics["auc"] > best_auc:
        best_auc = valid_metrics["auc"]
        torch.save(model.state_dict(), CONFIG["model_path"])
        print(f"⭐ Best Model Saved (Epoch {epoch+1}) AUC={best_auc:.4f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "train_auc": train_metrics["auc"],
        "valid_auc": valid_metrics["auc"],
        "valid_acc": valid_metrics["accuracy"],
        "valid_precision": valid_metrics["precision"],
        "valid_recall": valid_metrics["recall"],
        "valid_f1": valid_metrics["f1"],
        "valid_mcc": valid_metrics["mcc"],
        "valid_specificity": valid_metrics["specificity"],
        "lr": current_lr,
        "epoch_time": epoch_time,
    })

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch {epoch+1:03d}/{CONFIG['iteration']} | "
            f"Train AUC: {train_metrics['auc']:.4f} | "
            f"Valid AUC: {valid_metrics['auc']:.4f} | "
            f"Train Loss: {train_loss:.3f} | "
            f"Valid Loss: {valid_loss:.3f}"
        )

history_df = pd.DataFrame(history)
history_df.to_csv(CONFIG["log_path"], index=False)

training_time = (time.time() - training_start) / 60
print("\nTraining Finished!")
print(f"Total Time : {training_time:.2f} minutes")
print("Best Validation AUC:", round(best_auc, 4))

print("\n" + "=" * 70)
print("Loading Best v5 Model")
print("=" * 70)

model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
model.eval()
print("Best v5 model loaded successfully!")

print("\nRunning Final Test Evaluation...\n")
_, test_loss, test_predictions = tester.test_classifier(test_dataset)
test_metrics = evaluate(test_predictions)

print("=" * 70)
print("V5 FULL-RUN FINAL RESULTS")
print("=" * 70)
for metric, value in test_metrics.items():
    print(f"{metric:<15}: {value:.4f}")
print(f"{'test_loss':<15}: {test_loss:.4f}")

y_true, y_pred, y_score = test_predictions[0], test_predictions[1], test_predictions[2]
print(f"\nRaw score stats (test): min={y_score.min():.4f} max={y_score.max():.4f} "
      f"mean={y_score.mean():.4f} std={y_score.std():.4f}")

prediction_df = pd.DataFrame({
    "True_Label": y_true, "Predicted_Label": y_pred, "Prediction_Score": y_score,
})
prediction_df.to_csv(RESULTS_DIR / f"{DATASET}_v5_test_predictions.csv", index=False)

summary = {
    "Dataset": CONFIG["dataset"],
    "Epochs": CONFIG["iteration"],
    "Best Validation AUC": best_auc,
    "Final Test AUC": test_metrics["auc"],
    "Final Test Accuracy": test_metrics["accuracy"],
    "Final Test Precision": test_metrics["precision"],
    "Final Test Recall": test_metrics["recall"],
    "Final Test F1": test_metrics["f1"],
    "Final Test MCC": test_metrics["mcc"],
    "Final Test Specificity": test_metrics["specificity"],
}
pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{DATASET}_v5_summary.csv", index=False)

print("\n" + "=" * 70)
print("V5 Full Run Complete.")
print("=" * 70)
