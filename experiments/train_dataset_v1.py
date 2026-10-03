import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# ==========================================================
# Add Original D-GCAN
# ==========================================================

sys.path.append("/content/D-GCAN/DGCAN")

import preprocess as pp
from DGCAN import (
    MolecularGraphNeuralNetwork,
    Trainer,
    Tester
)

# ==========================================================
# Configuration
# ==========================================================

# ==========================================================
# Dataset Selection
# ==========================================================

DATASET = "BBBP"

BASE_PATH = "/content/drive/MyDrive/DGCAN_Project/datasets/processed"

CONFIG = {

    "dataset": DATASET,

    "train_file":
        f"{BASE_PATH}/{DATASET}_train.txt",

    "valid_file":
        f"{BASE_PATH}/{DATASET}_valid.txt",

    "test_file":
        f"{BASE_PATH}/{DATASET}_test.txt",

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

    "iteration": 2,

    "N": 5000,

    "threshold": 0.15,

    "seed": 42,

    "model_path":
        f"/content/drive/MyDrive/DGCAN_Project/models/{DATASET}_best_model.pth",

    "log_path":
        f"/content/drive/MyDrive/DGCAN_Project/results/{DATASET}_training_log.csv"

}
# ==========================================================
# Reproducibility
# ==========================================================

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# Create folders
# ==========================================================

os.makedirs(
    "/content/drive/MyDrive/DGCAN_Project/models",
    exist_ok=True
)

os.makedirs(
    "/content/drive/MyDrive/DGCAN_Project/results",
    exist_ok=True
)

# ==========================================================
# Evaluation Function
# ==========================================================

def evaluate(predictions):

    y_true = predictions[0]
    y_pred = predictions[1]
    y_score = predictions[2]

    auc = roc_auc_score(
        y_true,
        y_score
    )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    specificity = tn / (tn + fp)

    return {

        "auc": auc,

        "accuracy": acc,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "mcc": mcc,

        "specificity": specificity

    }


# ==========================================================
# Load Dataset
# ==========================================================

print("="*70)
print("Cross Dataset D-GCAN Framework")
print("="*70)

print("Dataset :", CONFIG["dataset"])
print("Device  :", device)

print("\nLoading Dataset...\n")

train_dataset = pp.create_dataset(
    CONFIG["train_file"],
    "",
    ""
)

valid_dataset = pp.create_dataset(
    CONFIG["valid_file"],
    "",
    ""
)

test_dataset = pp.create_dataset(
    CONFIG["test_file"],
    "",
    ""
)

print("Train :", len(train_dataset))
print("Valid :", len(valid_dataset))
print("Test  :", len(test_dataset))

# ==========================================================
# Create Model
# ==========================================================

print("\nCreating Model...\n")

model = MolecularGraphNeuralNetwork(

    CONFIG["N"],

    CONFIG["dim"],

    CONFIG["layer_hidden"],

    CONFIG["layer_output"],

    CONFIG["dropout"]

).to(device)

trainer = Trainer(

    model,

    CONFIG["lr"],

    CONFIG["batch_train"]

)

tester = Tester(

    model,

    CONFIG["batch_test"]

)

print("Model Ready")

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

print("="*70)

# ==========================================================
# Training Loop
# ==========================================================

history = []

best_auc = -1.0

print("\nStarting Training...\n")

training_start = time.time()

for epoch in range(CONFIG["iteration"]):

    # ------------------------------------------------------
    # Learning Rate Decay
    # ------------------------------------------------------

    if (
        (epoch + 1) % CONFIG["decay_interval"] == 0
    ):

        trainer.optimizer.param_groups[0]["lr"] *= \
            CONFIG["lr_decay"]

    current_lr = trainer.optimizer.param_groups[0]["lr"]

    epoch_start = time.time()

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    train_auc, train_loss, train_predictions = \
        trainer.train(train_dataset)

    train_metrics = evaluate(train_predictions)

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    _, valid_loss, valid_predictions = \
        tester.test_classifier(valid_dataset)

    valid_metrics = evaluate(valid_predictions)

    epoch_time = time.time() - epoch_start

    # ------------------------------------------------------
    # Save Best Model
    # ------------------------------------------------------

    if valid_metrics["auc"] > best_auc:

        best_auc = valid_metrics["auc"]

        torch.save(
            model.state_dict(),
            CONFIG["model_path"]
        )

        print(
            f"⭐ Best Model Saved "
            f"(Epoch {epoch+1}) "
            f"AUC={best_auc:.4f}"
        )

    # ------------------------------------------------------
    # Store History
    # ------------------------------------------------------

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

        "lr": current_lr,

        "epoch_time": epoch_time

    })

    # ------------------------------------------------------
    # Console Output
    # ------------------------------------------------------

    print(

        f"Epoch {epoch+1:03d}/{CONFIG['iteration']} | "

        f"Train AUC: {train_metrics['auc']:.4f} | "

        f"Valid AUC: {valid_metrics['auc']:.4f} | "

        f"Train Loss: {train_loss:.3f} | "

        f"Valid Loss: {valid_loss:.3f}"

    )

# ==========================================================
# Save Training History
# ==========================================================

history_df = pd.DataFrame(history)

history_df.to_csv(

    CONFIG["log_path"],

    index=False

)

training_time = (
    time.time() - training_start
) / 60

print("\nTraining Finished!")

print(
    f"Total Time : {training_time:.2f} minutes"
)

print(
    "Training Log Saved:"
)

print(
    CONFIG["log_path"]
)

print(
    "Best Validation AUC:",
    round(best_auc,4)
)

# ==========================================================
# Load Best Model
# ==========================================================

print("\n" + "=" * 70)
print("Loading Best Model")
print("=" * 70)

if os.path.exists(CONFIG["model_path"]):

    model.load_state_dict(
        torch.load(CONFIG["model_path"], map_location=device)
    )

    print("Best model loaded successfully!")

else:

    raise FileNotFoundError(
        f"Best model not found: {CONFIG['model_path']}"
    )

model.eval()

# ==========================================================
# Final Test Evaluation
# ==========================================================

print("\nRunning Final Test Evaluation...\n")

_, test_loss, test_predictions = tester.test_classifier(test_dataset)

test_metrics = evaluate(test_predictions)

print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

for metric, value in test_metrics.items():
    print(f"{metric:<15}: {value:.4f}")

print(f"{'test_loss':<15}: {test_loss:.4f}")

# ==========================================================
# Save Test Predictions
# ==========================================================

prediction_file = (
    f"/content/drive/MyDrive/DGCAN_Project/results/"
    f"{CONFIG['dataset']}_test_predictions.csv"
)

prediction_df = pd.DataFrame({

    "True_Label": test_predictions[0],

    "Predicted_Label": test_predictions[1],

    "Prediction_Score": test_predictions[2]

})

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\nPredictions Saved To:")
print(prediction_file)

# ==========================================================
# Save Summary
# ==========================================================

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

    "Final Test Specificity": test_metrics["specificity"]

}

summary_file = (
    f"/content/drive/MyDrive/DGCAN_Project/results/"
    f"{CONFIG['dataset']}_summary.csv"
)

pd.DataFrame([summary]).to_csv(
    summary_file,
    index=False
)

print("\nSummary Saved To:")
print(summary_file)

print("\n" + "=" * 70)
print("Experiment Completed Successfully!")
print("=" * 70)