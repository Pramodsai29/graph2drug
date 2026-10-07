"""
v12 — architecture comparison (roadmap step 4): GCN, GAT and GraphSAGE baselines,
to compare against the fixed D-GCAN (v8 BBBP / v9 BACE).

Same evaluation protocol as v8/v9 so the numbers are directly comparable:
  - identical scaffold-split files, skipping each file's first line exactly as
    D-GCAN's preprocess.create_dataset does -> the same molecules in every split
  - seeds 42-46, 140 epochs, best checkpoint by validation ROC-AUC
  - same metrics, same 0.15 classification threshold (AUC/PR-AUC are threshold-free)
Baselines use standard settings (not tuned): 3 message-passing layers, hidden 64,
ReLU, dropout 0.2, global mean pooling, linear head -> 2 logits, cross-entropy,
Adam lr 1e-3, batch 32. Atom features: RDKit one-hots (element, degree, formal
charge, total Hs, hybridisation) + aromatic + in-ring.

Runs on a Colab VM (/content) or locally. Set DATASET_OVERRIDE (and optionally
ARCHS_OVERRIDE, EPOCHS_OVERRIDE for a smoke test) in the kernel before
`colab exec -f`, or as environment variables locally.
"""
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix,
)
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/content") if Path("/content/datasets").exists() else Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def setting(name, default):
    return globals().get(name + "_OVERRIDE", os.environ.get(name, default))


DATASET = setting("DATASET", "BBBP")
ARCHS = setting("ARCHS", "GCN,GAT,SAGE")
ARCHS = ARCHS.split(",") if isinstance(ARCHS, str) else list(ARCHS)
EPOCHS = int(setting("EPOCHS", 140))
SEEDS = [42, 43, 44, 45, 46]
HIDDEN, LAYERS, DROPOUT, LR, BATCH = 64, 3, 0.2, 1e-3, 32
THRESHOLD = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- featurisation
ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 11, 19, 3, 12, 20, 26, 29, 30, 34]
HYB = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
       Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
       Chem.rdchem.HybridizationType.SP3D2]


def one_hot(value, choices):
    v = [0.0] * (len(choices) + 1)  # last slot = "other"
    v[choices.index(value) if value in choices else -1] = 1.0
    return v


def atom_features(a):
    return (one_hot(a.GetAtomicNum(), ELEMENTS) + one_hot(a.GetDegree(), [0, 1, 2, 3, 4, 5])
            + one_hot(a.GetFormalCharge(), [-1, 0, 1]) + one_hot(a.GetTotalNumHs(), [0, 1, 2, 3])
            + one_hot(a.GetHybridization(), HYB) + [float(a.GetIsAromatic()), float(a.IsInRing())])


def to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [(i, j), (j, i)]
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous() if edges
                  else torch.zeros((2, 0), dtype=torch.long))
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long), smiles=smiles)


def load_split(split):
    lines = (DATASET_DIR / f"{DATASET}_{split}.txt").read_text().strip().split("\n")[1:]
    graphs = [to_graph(ln.split()[0], int(ln.split()[1])) for ln in lines if ln.strip()]
    return [g for g in graphs if g is not None]


# ---------------------------------------------------------------- models
class GNN(nn.Module):
    def __init__(self, arch, in_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        for layer in range(LAYERS):
            d_in = in_dim if layer == 0 else HIDDEN
            if arch == "GCN":
                conv = GCNConv(d_in, HIDDEN)
            elif arch == "GAT":
                conv = GATConv(d_in, HIDDEN // 4, heads=4)  # 4 heads x 16 = 64
            elif arch == "SAGE":
                conv = SAGEConv(d_in, HIDDEN)
            else:
                raise ValueError(arch)
            self.convs.append(conv)
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, data):
        x = data.x
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, data.edge_index)), DROPOUT, self.training)
        return self.head(global_mean_pool(x, data.batch))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        ps.append(torch.softmax(model(batch), dim=1)[:, 1].cpu())
        ys.append(batch.y.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def evaluate(y, p):
    pred = (p > THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, pred), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "mcc": matthews_corrcoef(y, pred),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0}


# ---------------------------------------------------------------- run
print("=" * 70)
print(f"v12 architecture comparison — {DATASET} | {', '.join(ARCHS)} | {EPOCHS} epochs | {device}")
print("=" * 70)
train_set, valid_set, test_set = load_split("train"), load_split("valid"), load_split("test")
print("Train:", len(train_set), "Valid:", len(valid_set), "Test:", len(test_set))
in_dim = train_set[0].x.shape[1]
valid_loader = DataLoader(valid_set, batch_size=256)
test_loader = DataLoader(test_set, batch_size=256)

for arch in ARCHS:
    tag = f"v12_{arch}"
    rows, t_arch = [], time.time()
    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        gen = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, generator=gen)
        model = GNN(arch, in_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        path = MODELS_DIR / f"{DATASET}_{tag}_seed{seed}_best.pth"
        best_auc, best_epoch, log, t0 = -1.0, 0, [], time.time()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total = 0.0
            for batch in train_loader:
                batch = batch.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(batch), batch.y)
                loss.backward()
                opt.step()
                total += loss.item() * batch.num_graphs
            yv, pv = predict(model, valid_loader)
            v_auc = roc_auc_score(yv, pv)
            log.append({"epoch": epoch, "train_loss": total / len(train_set), "valid_auc": v_auc})
            if v_auc > best_auc:
                best_auc, best_epoch = v_auc, epoch
                torch.save(model.state_dict(), path)
        model.load_state_dict(torch.load(path, map_location=device))
        yt, pt = predict(model, test_loader)
        m = evaluate(yt, pt)
        pd.DataFrame(log).to_csv(RESULTS_DIR / f"{DATASET}_{tag}_seed{seed}_training_log.csv", index=False)
        pd.DataFrame({"y_true": yt, "y_pred": (pt > THRESHOLD).astype(int), "y_score": pt}).to_csv(
            RESULTS_DIR / f"{DATASET}_{tag}_seed{seed}_test_predictions.csv", index=False)
        minutes = (time.time() - t0) / 60
        print(f"  -> {arch} seed {seed}: best_valid_auc={best_auc:.4f} (epoch {best_epoch}) | "
              f"test_auc={m['auc']:.4f} | test_mcc={m['mcc']:.4f} | time={minutes:.2f} min")
        rows.append({"seed": seed, "best_valid_auc": best_auc, "best_epoch": best_epoch,
                     **{f"test_{k}": v for k, v in m.items()}, "minutes": minutes})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / f"{DATASET}_{tag}_multiseed_results.csv", index=False)
    summary = {"Dataset": DATASET, "Arch": arch, "N_seeds": len(SEEDS), "Epochs": EPOCHS,
               "Total_minutes": (time.time() - t_arch) / 60,
               "n_params": sum(p.numel() for p in model.parameters())}
    for k in ["auc", "pr_auc", "balanced_accuracy", "accuracy", "precision", "recall", "f1", "mcc", "specificity"]:
        summary[f"test_{k}_mean"] = res[f"test_{k}"].mean()
        summary[f"test_{k}_std"] = res[f"test_{k}"].std()
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / f"{DATASET}_{tag}_summary.csv", index=False)
    print(f"{arch} test_auc        : {summary['test_auc_mean']:.4f} +/- {summary['test_auc_std']:.4f}\n")
