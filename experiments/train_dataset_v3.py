import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
MODEL_PATH = PROJECT_ROOT / "models" / f"{DATASET}_best_model.pth"
TRAINING_LOG = PROJECT_ROOT / "results" / f"{DATASET}_training_log.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "radius": 1,
    "dim": 52,
    "layer_hidden": 4,
    "layer_output": 10,
    "dropout": 0.45,
    "batch_test": 8,
    "batch_train": 8,
    "N": 5000,
    "threshold": 0.15,
    "seed": 42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pp.device = device

print("=" * 70)
print("D-GCAN BBBP Root-Cause Diagnostics (v3) — inference/analysis only")
print("=" * 70)
print("Device:", device)

# ==========================================================
# Step 1 — Training curve analysis (uses existing v1 log, no rerun)
# ==========================================================

print("\n" + "-" * 70)
print("STEP 1: Training curve analysis (from BBBP_training_log.csv)")
print("-" * 70)

if TRAINING_LOG.exists():
    log_df = pd.read_csv(TRAINING_LOG)

    best_row = log_df.loc[log_df["valid_auc"].idxmax()]
    print("Best epoch by valid_auc:", int(best_row["epoch"]))
    print("  train_auc:", round(best_row["train_auc"], 4))
    print("  valid_auc:", round(best_row["valid_auc"], 4))
    print("  train_loss:", round(best_row["train_loss"], 4))
    print("  valid_loss:", round(best_row["valid_loss"], 4))
    print("First epoch valid_auc:", round(log_df["valid_auc"].iloc[0], 4))
    print("Last epoch valid_auc :", round(log_df["valid_auc"].iloc[-1], 4))
    print("Max valid_auc ever   :", round(log_df["valid_auc"].max(), 4))
    print("Min valid_auc ever   :", round(log_df["valid_auc"].min(), 4))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    axes[0].plot(log_df["epoch"], log_df["valid_loss"], label="valid_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss over training")
    axes[0].legend()

    axes[1].plot(log_df["epoch"], log_df["train_auc"], label="train_auc")
    axes[1].plot(log_df["epoch"], log_df["valid_auc"], label="valid_auc")
    axes[1].axhline(0.5, color="gray", linestyle="--", label="random (0.5)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC over training")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{DATASET}_v3_training_curves.png", dpi=150)
    plt.close()
else:
    print("Training log not found, skipping:", TRAINING_LOG)
    log_df = None

# ==========================================================
# Load datasets
# ==========================================================

print("\nLoading datasets...")
train_dataset = pp.create_dataset(str(TRAIN_FILE), "", "")
valid_dataset = pp.create_dataset(str(VALID_FILE), "", "")
test_dataset = pp.create_dataset(str(TEST_FILE), "", "")
print("Train:", len(train_dataset), "Valid:", len(valid_dataset), "Test:", len(test_dataset))

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
print("Best model loaded.")

# ==========================================================
# Step 2 — Final classifier layer (W_property) weight/bias inspection
# ==========================================================

print("\n" + "-" * 70)
print("STEP 2: Final layer (W_property) weight/bias stats")
print("-" * 70)

w = model.W_property.weight.detach().cpu().numpy()
b = model.W_property.bias.detach().cpu().numpy()

print("W_property.weight shape:", w.shape)
print("  mean:", w.mean(), " std:", w.std(), " min:", w.min(), " max:", w.max())
print("W_property.bias:", b)

layer_stats = {
    "weight_mean": float(w.mean()),
    "weight_std": float(w.std()),
    "weight_min": float(w.min()),
    "weight_max": float(w.max()),
    "bias_class0": float(b[0]),
    "bias_class1": float(b[1]),
    "bias_diff_1_minus_0": float(b[1] - b[0]),
}
pd.DataFrame([layer_stats]).to_csv(RESULTS_DIR / f"{DATASET}_v3_final_layer_stats.csv", index=False)

# ==========================================================
# Helper: replicate mlp() but return PRE-sigmoid logits too
# ==========================================================

def forward_with_logits(model, inputs):
    """Mirrors model.gnn() + model.mlp() but also returns the raw
    pre-sigmoid logits from W_property, which mlp() normally discards."""
    Smiles, molecular_vectors = model.gnn(inputs)
    vectors = molecular_vectors
    for l in range(model.layer_output):
        vectors = torch.relu(model.W_output[l](vectors))
    raw_logits = model.W_property(vectors)          # pre-sigmoid, shape [batch, 2]
    sigmoid_out = torch.sigmoid(raw_logits)          # what mlp() actually returns
    return Smiles, raw_logits, sigmoid_out

# ==========================================================
# Step 3 — Raw pre-sigmoid logits on real test molecules
# ==========================================================

print("\n" + "-" * 70)
print("STEP 3: Pre-sigmoid logits on first 20 test molecules")
print("-" * 70)

sample = test_dataset[:20]
data_batch = list(zip(*sample))
inputs = data_batch[:-1]

with torch.no_grad():
    Smiles, raw_logits, sigmoid_out = forward_with_logits(model, inputs)

raw_logits_np = raw_logits.cpu().numpy()
sigmoid_out_np = sigmoid_out.cpu().numpy()

print("Raw logits (class0, class1) — first 20:")
for i in range(len(Smiles)):
    print(f"  {Smiles[i][:30]:<30}  logit0={raw_logits_np[i,0]:8.3f}  logit1={raw_logits_np[i,1]:8.3f}  "
          f"sigmoid0={sigmoid_out_np[i,0]:.6f}  sigmoid1={sigmoid_out_np[i,1]:.6f}")

print("\nLogit class1 stats: min={:.3f} max={:.3f} mean={:.3f} std={:.3f}".format(
    raw_logits_np[:, 1].min(), raw_logits_np[:, 1].max(),
    raw_logits_np[:, 1].mean(), raw_logits_np[:, 1].std()
))

pd.DataFrame({
    "smiles": Smiles,
    "logit_class0": raw_logits_np[:, 0],
    "logit_class1": raw_logits_np[:, 1],
    "sigmoid_class0": sigmoid_out_np[:, 0],
    "sigmoid_class1": sigmoid_out_np[:, 1],
}).to_csv(RESULTS_DIR / f"{DATASET}_v3_logit_samples.csv", index=False)

# ==========================================================
# Step 4 — Input perturbation test: does output move at all?
# ==========================================================

print("\n" + "-" * 70)
print("STEP 4: Input perturbation test (real vs. randomized graph input)")
print("-" * 70)

perturbation_rows = []

for idx in range(3):
    smiles, fingerprints, adjacency, mol_size, prop = test_dataset[idx]

    # Original single-molecule batch
    orig_batch = list(zip(*[test_dataset[idx]]))
    orig_inputs = orig_batch[:-1]

    # Randomized fingerprints (random vocab ids) and randomized adjacency (same shape)
    rand_fingerprints = torch.randint(
        0, CONFIG["N"], fingerprints.shape, dtype=fingerprints.dtype, device=device
    )
    rand_adjacency = (torch.rand(adjacency.shape, device=device) > 0.5).float()

    perturbed_entry = (smiles, rand_fingerprints, rand_adjacency, mol_size, prop)
    pert_batch = list(zip(*[perturbed_entry]))
    pert_inputs = pert_batch[:-1]

    with torch.no_grad():
        _, orig_logits, orig_sigmoid = forward_with_logits(model, orig_inputs)
        _, pert_logits, pert_sigmoid = forward_with_logits(model, pert_inputs)

    orig_l1 = orig_logits.cpu().numpy()[0, 1]
    pert_l1 = pert_logits.cpu().numpy()[0, 1]
    orig_s1 = orig_sigmoid.cpu().numpy()[0, 1]
    pert_s1 = pert_sigmoid.cpu().numpy()[0, 1]

    print(f"Molecule {idx} ({smiles[:30]}):")
    print(f"  original : logit1={orig_l1:.3f}  sigmoid1={orig_s1:.6f}")
    print(f"  perturbed: logit1={pert_l1:.3f}  sigmoid1={pert_s1:.6f}")
    print(f"  |delta logit1| = {abs(orig_l1 - pert_l1):.4f}")

    perturbation_rows.append({
        "smiles": smiles,
        "orig_logit1": orig_l1,
        "pert_logit1": pert_l1,
        "delta_logit1": orig_l1 - pert_l1,
        "orig_sigmoid1": orig_s1,
        "pert_sigmoid1": pert_s1,
    })

pd.DataFrame(perturbation_rows).to_csv(
    RESULTS_DIR / f"{DATASET}_v3_perturbation_test.csv", index=False
)

# ==========================================================
# Step 5 — Score distribution across train / valid / test splits
# ==========================================================

print("\n" + "-" * 70)
print("STEP 5: Prediction score distribution across splits")
print("-" * 70)

tester = Tester(model, CONFIG["batch_test"])

split_stats = []
split_scores = {}

for split_name, split_data in [("train", train_dataset), ("valid", valid_dataset), ("test", test_dataset)]:
    with torch.no_grad():
        _, _, predictions = tester.test_classifier(split_data)
    y_score = predictions[2].astype(float)
    split_scores[split_name] = y_score
    stats = {
        "split": split_name,
        "count": len(y_score),
        "min": float(np.min(y_score)),
        "max": float(np.max(y_score)),
        "mean": float(np.mean(y_score)),
        "std": float(np.std(y_score)),
    }
    split_stats.append(stats)
    print(f"{split_name:<6}: n={stats['count']:<5} min={stats['min']:.6f} max={stats['max']:.6f} "
          f"mean={stats['mean']:.6f} std={stats['std']:.2e}")

pd.DataFrame(split_stats).to_csv(RESULTS_DIR / f"{DATASET}_v3_split_score_stats.csv", index=False)

plt.figure(figsize=(8, 5))
for split_name, scores in split_scores.items():
    plt.hist(scores, bins=30, alpha=0.5, label=split_name)
plt.axvline(CONFIG["threshold"], color="red", linestyle="--", label=f"threshold={CONFIG['threshold']}")
plt.xlabel("Raw prediction score")
plt.ylabel("Count")
plt.title(f"{DATASET} — Prediction Score Distribution by Split")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"{DATASET}_v3_split_score_histograms.png", dpi=150)
plt.close()

# ==========================================================
# Step 6 — One-shot gradient diagnostic (no optimizer step, no weight update)
# ==========================================================

print("\n" + "-" * 70)
print("STEP 6: Gradient diagnostic (single batch, no weight update)")
print("-" * 70)

model.train()
model.zero_grad()

grad_batch = list(zip(*train_dataset[:CONFIG["batch_train"]]))
_, loss, _, _ = model.forward_classifier(grad_batch, train=True)
loss.backward()

gradient_rows = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        gradient_rows.append({"layer": name, "grad_norm": grad_norm})

grad_df = pd.DataFrame(gradient_rows)
print(grad_df.to_string(index=False))

grad_df.to_csv(RESULTS_DIR / f"{DATASET}_v3_gradient_norms.csv", index=False)

model.zero_grad()
model.eval()

print(f"\nSingle-batch train-mode loss (F.cross_entropy on sigmoid outputs): {loss.item():.4f}")

# ==========================================================
# Summary
# ==========================================================

summary_lines = [
    "=" * 70,
    "V3 DIAGNOSTIC SUMMARY",
    "=" * 70,
    "",
    "CODE-LEVEL OBSERVATION (DGCAN.py, MolecularGraphNeuralNetwork.mlp/forward_classifier):",
    "  mlp() applies torch.sigmoid() to the 2-class output of W_property,",
    "  then forward_classifier() passes that sigmoid output directly into",
    "  F.cross_entropy(), which internally applies log_softmax. This means",
    "  the loss is computed on softmax(sigmoid(logits)) instead of the raw",
    "  logits cross_entropy expects — a compounded/incorrect nonlinearity ",
    "  stack. This is upstream code (vendor/D-GCAN), not something we wrote.",
    "",
    f"Logit class1 range on sampled test molecules: "
    f"[{raw_logits_np[:,1].min():.3f}, {raw_logits_np[:,1].max():.3f}]",
    f"Final layer bias: class0={b[0]:.4f}, class1={b[1]:.4f} (diff={b[1]-b[0]:.4f})",
    "",
    "See per-step CSV/PNG outputs in results/ for full numeric evidence.",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(RESULTS_DIR / f"{DATASET}_v3_diagnostic_summary.txt", "w") as f:
    f.write(summary_text + "\n")

print("\nAll v3 outputs written to:", RESULTS_DIR)
