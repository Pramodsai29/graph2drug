import os
import sys
import time
import types
import random
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
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "vendor" / "D-GCAN" / "DGCAN"))

import preprocess as pp
from DGCAN import MolecularGraphNeuralNetwork, Trainer, Tester

DATASET = "BBBP"

BASE_PATH = PROJECT_ROOT / "datasets" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "dataset": DATASET,
    "train_file": str(BASE_PATH / f"{DATASET}_train.txt"),
    "valid_file": str(BASE_PATH / f"{DATASET}_valid.txt"),
    "test_file": str(BASE_PATH / f"{DATASET}_test.txt"),
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
    # SMOKE TEST: short run to verify the fix works before a full 140-epoch retrain.
    "iteration": 10,
    "N": 5000,
    "threshold": 0.15,
    "seed": 42,
    "model_path": str(MODELS_DIR / f"{DATASET}_v4_best_model.pth"),
    "log_path": str(RESULTS_DIR / f"{DATASET}_v4_training_log.csv"),
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp.device = device

print("=" * 70)
print("D-GCAN v4 — FIX: raw logits -> cross_entropy (no sigmoid-before-CE)")
print("SMOKE TEST:", CONFIG["iteration"], "epochs (not the full 140-epoch run)")
print("=" * 70)
print("Dataset:", CONFIG["dataset"])
print("Device :", device)

# ==========================================================
# The fix: replace forward_classifier's sigmoid-before-cross_entropy
# with raw logits -> cross_entropy, and softmax only for reporting.
# Applied via monkey-patch on the model INSTANCE so vendor/D-GCAN/DGCAN.py
# stays untouched (read-only upstream code, same rule as train_dataset_v1.py).
# ==========================================================

def fixed_forward_classifier(self, data_batch, train):
    inputs = data_batch[:-1]
    correct_labels = torch.cat(data_batch[-1])

    if train:
        Smiles, molecular_vectors = self.gnn(inputs)
        vectors = molecular_vectors
        for l in range(self.layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        logits = self.W_property(vectors)              # raw logits (FIX: no sigmoid here)
        loss = F.cross_entropy(logits, correct_labels)  # correct usage now
        probs = F.softmax(logits, dim=1)                # for score reporting only
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


# ==========================================================
# Step 0 — Before/after gradient sanity check on a fresh model
# (confirms the fix restores gradient flow before spending time training)
# ==========================================================

print("\n" + "-" * 70)
print("STEP 0: Gradient sanity check (original buggy loss vs. fixed loss)")
print("        same freshly-initialized model, same batch, no weight update")
print("-" * 70)

train_dataset = pp.create_dataset(CONFIG["train_file"], "", "")

sanity_model = MolecularGraphNeuralNetwork(
    CONFIG["N"], CONFIG["dim"], CONFIG["layer_hidden"], CONFIG["layer_output"], CONFIG["dropout"]
).to(device)

sanity_batch = list(zip(*train_dataset[: CONFIG["batch_train"]]))

# Original (buggy) loss, via the unpatched class method
sanity_model.zero_grad()
_, orig_loss, _, _ = MolecularGraphNeuralNetwork.forward_classifier(sanity_model, sanity_batch, train=True)
orig_loss.backward()
orig_grad_norms = [p.grad.norm().item() for p in sanity_model.parameters() if p.grad is not None]
sanity_model.zero_grad()

# Fixed loss, same model, same batch
sanity_model.forward_classifier = types.MethodType(fixed_forward_classifier, sanity_model)
_, fixed_loss, _, _ = sanity_model.forward_classifier(sanity_batch, train=True)
fixed_loss.backward()
fixed_grad_norms = [p.grad.norm().item() for p in sanity_model.parameters() if p.grad is not None]
sanity_model.zero_grad()

print(f"Original loss: {orig_loss.item():.4f}  | mean grad norm: {np.mean(orig_grad_norms):.3e}  "
      f"| max grad norm: {np.max(orig_grad_norms):.3e}")
print(f"Fixed loss   : {fixed_loss.item():.4f}  | mean grad norm: {np.mean(fixed_grad_norms):.3e}  "
      f"| max grad norm: {np.max(fixed_grad_norms):.3e}")
print(f"Mean grad norm improved by ~{np.mean(fixed_grad_norms) / max(np.mean(orig_grad_norms), 1e-30):.1f}x")

del sanity_model

# ==========================================================
# Actual short training run with a fresh model (fix applied from the start)
# ==========================================================

print("\nLoading datasets...\n")

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


history = []
best_auc = -1.0

print("\nStarting Training (smoke test)...\n")
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
print("\nSmoke Test Training Finished!")
print(f"Total Time : {training_time:.2f} minutes")
print("Best Validation AUC:", round(best_auc, 4))

# ==========================================================
# Load best model, final test evaluation
# ==========================================================

print("\n" + "=" * 70)
print("Loading Best v4 Model")
print("=" * 70)

if not os.path.exists(CONFIG["model_path"]):
    raise FileNotFoundError(f"Best model not found: {CONFIG['model_path']}")

model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
model.eval()
print("Best v4 model loaded successfully!")

print("\nRunning Final Test Evaluation...\n")
_, test_loss, test_predictions = tester.test_classifier(test_dataset)
test_metrics = evaluate(test_predictions)

print("=" * 70)
print("V4 SMOKE-TEST FINAL RESULTS")
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
prediction_df.to_csv(RESULTS_DIR / f"{DATASET}_v4_test_predictions.csv", index=False)

summary = {
    "Dataset": CONFIG["dataset"],
    "Epochs": CONFIG["iteration"],
    "Note": "SMOKE TEST - short run to validate the sigmoid-before-cross_entropy fix",
    "Best Validation AUC": best_auc,
    "Final Test AUC": test_metrics["auc"],
    "Final Test Accuracy": test_metrics["accuracy"],
    "Final Test Precision": test_metrics["precision"],
    "Final Test Recall": test_metrics["recall"],
    "Final Test F1": test_metrics["f1"],
    "Final Test MCC": test_metrics["mcc"],
    "Final Test Specificity": test_metrics["specificity"],
}
pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{DATASET}_v4_summary.csv", index=False)

print("\n" + "=" * 70)
print("V4 Smoke Test Complete.")
print("If valid_auc/test_auc moved meaningfully above 0.5, the fix is validated")
print("and a full-length retrain (v5, 140 epochs) can be run next.")
print("=" * 70)
