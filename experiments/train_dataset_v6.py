import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "vendor" / "D-GCAN" / "DGCAN"))

import preprocess as pp
from DGCAN import MolecularGraphNeuralNetwork, Tester

DATASET = "BBBP"

TRAIN_FILE = PROJECT_ROOT / "datasets" / "processed" / f"{DATASET}_train.txt"
VALID_FILE = PROJECT_ROOT / "datasets" / "processed" / f"{DATASET}_valid.txt"
TEST_FILE = PROJECT_ROOT / "datasets" / "processed" / f"{DATASET}_test.txt"
MODEL_PATH = PROJECT_ROOT / "models" / f"{DATASET}_v5_best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "radius": 1, "dim": 52, "layer_hidden": 4, "layer_output": 10,
    "dropout": 0.45, "batch_test": 8, "N": 5000, "seed": 42,
    "n_bootstrap": 2000,
}

np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp.device = device

print("=" * 70)
print("D-GCAN v6 — Investigating the validation/test AUC gap (v5 model)")
print("=" * 70)

# Same fix as v4/v5, needed so Tester.test_classifier uses the corrected loss/scoring
def fixed_forward_classifier(self, data_batch, train):
    inputs = data_batch[:-1]
    correct_labels = torch.cat(data_batch[-1])
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
# Load datasets + model
# ==========================================================

print("\nLoading datasets...")
train_dataset = pp.create_dataset(str(TRAIN_FILE), "", "")
valid_dataset = pp.create_dataset(str(VALID_FILE), "", "")
test_dataset = pp.create_dataset(str(TEST_FILE), "", "")
print("Train:", len(train_dataset), "Valid:", len(valid_dataset), "Test:", len(test_dataset))

model = MolecularGraphNeuralNetwork(
    CONFIG["N"], CONFIG["dim"], CONFIG["layer_hidden"], CONFIG["layer_output"], CONFIG["dropout"]
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.forward_classifier = types.MethodType(fixed_forward_classifier, model)
tester = Tester(model, CONFIG["batch_test"])
print("v5 best model loaded.")

# ==========================================================
# Step A — Bootstrap AUC confidence intervals (valid vs test)
# ==========================================================

print("\n" + "-" * 70)
print("STEP A: Bootstrap 95% CI for AUC (valid vs. test, same v5 model)")
print("-" * 70)

def bootstrap_auc_ci(y_true, y_score, n_boot, seed):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return {
        "point_auc": roc_auc_score(y_true, y_score),
        "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
        "boot_mean": float(aucs.mean()),
        "boot_std": float(aucs.std()),
    }

with torch.no_grad():
    _, _, valid_predictions = tester.test_classifier(valid_dataset)
    _, _, test_predictions = tester.test_classifier(test_dataset)

valid_y_true = valid_predictions[0].astype(int)
valid_y_score = valid_predictions[2].astype(float)
test_y_true = test_predictions[0].astype(int)
test_y_score = test_predictions[2].astype(float)

valid_ci = bootstrap_auc_ci(valid_y_true, valid_y_score, CONFIG["n_bootstrap"], CONFIG["seed"])
test_ci = bootstrap_auc_ci(test_y_true, test_y_score, CONFIG["n_bootstrap"], CONFIG["seed"] + 1)

print(f"Valid AUC: {valid_ci['point_auc']:.4f}  95% CI [{valid_ci['ci_low']:.4f}, {valid_ci['ci_high']:.4f}]  "
      f"(n={len(valid_y_true)})")
print(f"Test  AUC: {test_ci['point_auc']:.4f}  95% CI [{test_ci['ci_low']:.4f}, {test_ci['ci_high']:.4f}]  "
      f"(n={len(test_y_true)})")

overlap = not (test_ci["ci_high"] < valid_ci["ci_low"] or valid_ci["ci_high"] < test_ci["ci_low"])
print(f"\nCIs overlap: {overlap}")
if overlap:
    print("  -> Some of the gap could plausibly be sampling noise, but check the CI widths below --")
    print("     note this model is evaluated on the CURRENT v5 checkpoint (epoch 16), not on the")
    print("     original noisy epoch-argmax valid_auc=0.9563 reported live during training (see Step B).")
else:
    print("  -> Gap is unlikely to be explained by sampling noise alone.")

pd.DataFrame([
    {"split": "valid", **valid_ci, "n": len(valid_y_true)},
    {"split": "test", **test_ci, "n": len(test_y_true)},
]).to_csv(RESULTS_DIR / f"{DATASET}_v6_bootstrap_auc.csv", index=False)

# ==========================================================
# Step B — Was epoch 16 a noisy spike? Re-examine the training log
# ==========================================================

print("\n" + "-" * 70)
print("STEP B: Was the selected checkpoint (epoch 16) a validation-AUC spike?")
print("-" * 70)

log_path = RESULTS_DIR / f"{DATASET}_v5_training_log.csv"
if log_path.exists():
    log_df = pd.read_csv(log_path)
    window = log_df[(log_df["epoch"] >= 10) & (log_df["epoch"] <= 25)]
    print("valid_auc, epochs 10-25 (selected epoch=16):")
    print(window[["epoch", "valid_auc"]].to_string(index=False))
    neighbor_mean = log_df[(log_df["epoch"] >= 10) & (log_df["epoch"] <= 22) & (log_df["epoch"] != 16)]["valid_auc"].mean()
    epoch16_auc = log_df[log_df["epoch"] == 16]["valid_auc"].values[0]
    print(f"\nEpoch 16 valid_auc: {epoch16_auc:.4f}")
    print(f"Mean valid_auc of neighboring epochs (10-22, excl. 16): {neighbor_mean:.4f}")
    print(f"Epoch 16 excess over neighbors: {epoch16_auc - neighbor_mean:+.4f}")
else:
    print("v5 training log not found, skipping.")

# ==========================================================
# Step C — Structural novelty: is test more distinct from train than valid is?
# ==========================================================

print("\n" + "-" * 70)
print("STEP C: Structural novelty (Morgan/ECFP4 Tanimoto similarity to train)")
print("-" * 70)

def mol_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

train_smiles = [d[0] for d in train_dataset]
valid_smiles = [d[0] for d in valid_dataset]
test_smiles = [d[0] for d in test_dataset]

train_fps = [fp for fp in (mol_fp(s) for s in train_smiles) if fp is not None]

def max_sim_to_train(smiles_list, train_fps):
    sims = []
    for s in smiles_list:
        fp = mol_fp(s)
        if fp is None:
            continue
        sim = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        sims.append(sim)
    return np.array(sims)

valid_max_sim = max_sim_to_train(valid_smiles, train_fps)
test_max_sim = max_sim_to_train(test_smiles, train_fps)

print(f"Valid-to-train max Tanimoto similarity: mean={valid_max_sim.mean():.4f} "
      f"median={np.median(valid_max_sim):.4f} std={valid_max_sim.std():.4f}")
print(f"Test-to-train  max Tanimoto similarity: mean={test_max_sim.mean():.4f} "
      f"median={np.median(test_max_sim):.4f} std={test_max_sim.std():.4f}")

diff = valid_max_sim.mean() - test_max_sim.mean()
print(f"\nValid is {'MORE' if diff > 0 else 'LESS'} similar to train than test is "
      f"(difference in mean max-similarity: {diff:+.4f})")

pd.DataFrame([
    {"split": "valid", "mean_max_sim_to_train": float(valid_max_sim.mean()),
     "median_max_sim_to_train": float(np.median(valid_max_sim)), "std": float(valid_max_sim.std()), "n": len(valid_max_sim)},
    {"split": "test", "mean_max_sim_to_train": float(test_max_sim.mean()),
     "median_max_sim_to_train": float(np.median(test_max_sim)), "std": float(test_max_sim.std()), "n": len(test_max_sim)},
]).to_csv(RESULTS_DIR / f"{DATASET}_v6_novelty_stats.csv", index=False)

plt.figure(figsize=(7, 5))
plt.hist(valid_max_sim, bins=25, alpha=0.6, label="valid (max sim to train)")
plt.hist(test_max_sim, bins=25, alpha=0.6, label="test (max sim to train)")
plt.xlabel("Max Tanimoto similarity to nearest train molecule")
plt.ylabel("Count")
plt.title(f"{DATASET} — Structural Novelty of Valid vs. Test Relative to Train")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"{DATASET}_v6_novelty_histogram.png", dpi=150)
plt.close()

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 70)
print("V6 SUMMARY")
print("=" * 70)
print(f"Valid AUC {valid_ci['point_auc']:.4f} [{valid_ci['ci_low']:.4f},{valid_ci['ci_high']:.4f}] vs. "
      f"Test AUC {test_ci['point_auc']:.4f} [{test_ci['ci_low']:.4f},{test_ci['ci_high']:.4f}]  "
      f"(overlap={overlap})")
print(f"Structural novelty gap (mean max-sim-to-train, valid - test): {diff:+.4f}")
print("\nOutputs written to results/: BBBP_v6_bootstrap_auc.csv, BBBP_v6_novelty_stats.csv, "
      "BBBP_v6_novelty_histogram.png")
