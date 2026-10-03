import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "vendor" / "D-GCAN" / "DGCAN"))

import preprocess as pp
from DGCAN import MolecularGraphNeuralNetwork, Tester

DATASET = "BBBP"

TEST_FILE = PROJECT_ROOT / "datasets" / "processed" / f"{DATASET}_test.txt"
MODEL_PATH = PROJECT_ROOT / "models" / f"{DATASET}_best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "radius": 1,
    "dim": 52,
    "layer_hidden": 4,
    "layer_output": 10,
    "dropout": 0.45,
    "batch_test": 8,
    "N": 5000,
    "threshold": 0.15,
    "seed": 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp.device = device  # preprocess.py hardcodes device='cuda' at import time

print("=" * 70)
print("D-GCAN BBBP Diagnostics (v2) — inference only, no training")
print("=" * 70)
print("Device      :", device)
print("Model file  :", MODEL_PATH)
print("Test file   :", TEST_FILE)

# ==========================================================
# Load test dataset
# ==========================================================

print("\nLoading test dataset...\n")

test_dataset = pp.create_dataset(str(TEST_FILE), "", "")

print("Test molecules:", len(test_dataset))

# ==========================================================
# Load model
# ==========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Best model not found: {MODEL_PATH}")

model = MolecularGraphNeuralNetwork(
    CONFIG["N"],
    CONFIG["dim"],
    CONFIG["layer_hidden"],
    CONFIG["layer_output"],
    CONFIG["dropout"],
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Best model loaded successfully.")

tester = Tester(model, CONFIG["batch_test"])

# ==========================================================
# Run inference (no gradients, no training)
# ==========================================================

with torch.no_grad():
    acc, loss_total, predictions = tester.test_classifier(test_dataset)

y_true = predictions[0].astype(int)
y_pred = predictions[1].astype(int)
y_score = predictions[2].astype(float)

# ==========================================================
# Raw prediction score stats
# ==========================================================

score_stats = {
    "count": len(y_score),
    "min": float(np.min(y_score)),
    "max": float(np.max(y_score)),
    "mean": float(np.mean(y_score)),
    "std": float(np.std(y_score)),
}

print("\n" + "=" * 70)
print("RAW PREDICTION SCORE STATS")
print("=" * 70)
for k, v in score_stats.items():
    print(f"{k:<10}: {v}")

print("\nFirst 20 raw scores:")
print(y_score[:20])

pd.DataFrame([score_stats]).to_csv(
    RESULTS_DIR / f"{DATASET}_v2_score_stats.csv", index=False
)

# ==========================================================
# Score histogram
# ==========================================================

plt.figure(figsize=(7, 5))
plt.hist(y_score, bins=30, color="steelblue", edgecolor="black")
plt.axvline(CONFIG["threshold"], color="red", linestyle="--", label=f"threshold={CONFIG['threshold']}")
plt.xlabel("Raw prediction score")
plt.ylabel("Count")
plt.title(f"{DATASET} Test Set — Raw Prediction Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"{DATASET}_v2_score_histogram.png", dpi=150)
plt.close()

# ==========================================================
# ROC curve
# ==========================================================

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"{DATASET} Test Set — ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"{DATASET}_v2_roc_curve.png", dpi=150)
plt.close()

print(f"\nROC AUC (recomputed): {roc_auc:.4f}")

# ==========================================================
# Confusion matrix
# ==========================================================

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion matrix (threshold = {:.2f}):".format(CONFIG["threshold"]))
print(cm)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

plt.figure(figsize=(5, 5))
plt.imshow(cm, cmap="Blues")
plt.title(f"{DATASET} Test Set — Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Pred 0", "Pred 1"])
plt.yticks([0, 1], ["True 0", "True 1"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"{DATASET}_v2_confusion_matrix.png", dpi=150)
plt.close()

# ==========================================================
# Save predictions + summary
# ==========================================================

pd.DataFrame({
    "True_Label": y_true,
    "Predicted_Label": y_pred,
    "Prediction_Score": y_score,
}).to_csv(RESULTS_DIR / f"{DATASET}_v2_test_predictions.csv", index=False)

summary = {
    "Dataset": DATASET,
    "Test_AUC_recomputed": roc_auc,
    "Score_min": score_stats["min"],
    "Score_max": score_stats["max"],
    "Score_mean": score_stats["mean"],
    "Score_std": score_stats["std"],
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
}
pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{DATASET}_v2_diagnostics_summary.csv", index=False)

print("\n" + "=" * 70)
print("Diagnostics complete. Outputs written to:", RESULTS_DIR)
print("=" * 70)
