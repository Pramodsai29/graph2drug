# Graph2Drug

A reproducible, cross-dataset framework for molecular property prediction built
on **D-GCAN** (Directed Graph Convolutional Attention Network). The project
reproduces D-GCAN, fixes a training bug in the original implementation, and
evaluates it under a statistically sound multi-seed protocol on four
MoleculeNet benchmarks: **BBBP, BACE, ClinTox and Tox21**.

> Status: work in progress toward a conference paper. Results below are final
> for the datasets marked complete; everything else is listed under
> [Roadmap](#roadmap).

---

## Key finding: the original D-GCAN classifier cannot learn

Training the upstream D-GCAN code on BBBP (scaffold split) gave a test AUC of
**0.466** — no better than chance — with every molecule predicted positive.

Diagnostics (`experiments/train_dataset_v2.py`, `v3.py`, inference-only) traced
this to a single root cause in the upstream `DGCAN.py`:

- `mlp()` applies `torch.sigmoid()` to the classifier output, and
  `forward_classifier()` then passes that **already-squashed** output into
  `F.cross_entropy()`, which applies its own softmax/log.
- Logits grow to ±20–80 within the first few epochs, the sigmoid saturates to
  exactly 0/1, and its derivative collapses — gradients in every layer fall to
  1e-10…1e-7.
- Validation AUC freezes at 0.5 from epoch 5 onward, and the output no longer
  responds to the input (replacing a molecule with random noise changes the
  logit by < 0.1).

**Fix:** pass raw logits to `F.cross_entropy` (applied as a monkey-patch of
`forward_classifier` on the model instance; upstream code is left untouched).
Mean gradient norm improves ~4× immediately, and validation AUC rises from
0.5 to ~0.95 within a few epochs.

| BBBP (single run) | Original (bug) | Fixed |
|---|---|---|
| Test AUC | 0.4662 | **0.6523** |
| Specificity | 0.0000 | 0.2247 |
| MCC | 0.0000 | 0.1435 |
| Prediction score std | ~8e-7 (collapsed) | 0.35 |

---

## Results — multi-seed baseline (fixed D-GCAN)

Protocol, identical for every dataset: DeepChem `ScaffoldSplitter` 80/10/10
(seed 42), original D-GCAN hyperparameters, 140 epochs, best checkpoint chosen
by validation AUC, **5 training seeds (42–46), reported as mean ± std**.
Tox21 uses the single SR-MMP task; ClinTox uses `CT_TOX`.

| Dataset | Test AUC | Test PR-AUC | Balanced Acc. | MCC | Specificity |
|---|---|---|---|---|---|
| BBBP | **0.6396 ± 0.0120** | — | — | 0.1213 ± 0.0371 | 0.1798 ± 0.0381 |
| BACE | **0.7938 ± 0.0191** | 0.8161 ± 0.0186 | 0.7397 ± 0.0216 | 0.4770 ± 0.0448 | 0.7233 ± 0.0596 |
| ClinTox (CT_TOX) | **0.8419 ± 0.0285** | 0.3409 ± 0.0714 | 0.7421 ± 0.0410 | 0.3315 ± 0.0618 | 0.8642 ± 0.0340 |
| Tox21 (SR-MMP) | *pending* | | | | |

Notes:
- **Training is reproducible**: seed-to-seed std of test AUC (≈0.01–0.03) is
  far smaller than the test-set sampling uncertainty (BBBP bootstrap 95% CI
  width ≈ 0.16 with n = 183).
- **BBBP's valid/test gap is dataset-specific.** On BBBP the best validation
  AUC (~0.95) is far above test AUC (~0.65); test molecules are measurably
  less similar to the training set than validation molecules (mean nearest-
  neighbour Tanimoto 0.415 vs 0.472). On BACE the gap reverses (valid ~0.74,
  test ~0.79), so this is not a generic scaffold-split artefact.
- **ClinTox is highly imbalanced** (10 positives in 144 test molecules), so its
  AUC is noisy; PR-AUC 0.34 is ~5× the 7% positive rate.
- A dropout / weight-decay sweep (`train_dataset_v7.py`) found no robust
  improvement over the original hyperparameters.

---

## Demo: predict a molecule

```bash
python experiments/demo_predict.py --dataset BBBP --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"   # caffeine
python experiments/demo_predict.py --dataset BBBP --verify    # reproduces the reported test AUC
```

Loads a trained checkpoint and predicts on CPU in ~10 s (`--dataset BBBP` or `BACE`).
See [DEMO.md](DEMO.md) for a full walkthrough.

---

## Repository layout

```
datasets/
  raw/          MoleculeNet CSVs (BBBP, BACE, ClinTox, Tox21)
  processed/    cleaned data + scaffold splits ({DATASET}_train/valid/test.txt)
experiments/
  prepare_dataset.py   cleaning + scaffold split for BACE / ClinTox / Tox21
  train_dataset_v1.py  original pipeline (reproduces the bug; frozen baseline)
  train_dataset_v2.py  diagnostics: saturated prediction scores
  train_dataset_v3.py  root-cause diagnostics (logits, gradients, perturbation)
  train_dataset_v4.py  fix + 10-epoch smoke test (CPU)
  train_dataset_v5.py  fix + full 140-epoch run (Colab GPU)
  train_dataset_v6.py  valid/test AUC gap investigation (bootstrap, novelty)
  train_dataset_v7.py  dropout / weight-decay sweep
  train_dataset_v8.py  5-seed BBBP baseline (official BBBP number)
  train_dataset_v9.py  5-seed baseline for BACE / ClinTox / Tox21
  demo_predict.py      predict any SMILES with a trained checkpoint (CPU)
models/         trained checkpoints (.pth)
results/        per-run CSVs, training logs, predictions and diagnostic plots
notebooks/      exploratory notebooks
Phase1/, Phase2/  earlier exploratory runs (not part of the reported results)
```

Each `train_dataset_vN.py` adds a capability without modifying earlier
versions, so every reported number can be traced to the script that produced
it.

---

## Reproducing

Requirements: Python 3.10+, `torch`, `rdkit`, `pandas`, `scikit-learn`,
`matplotlib`; `deepchem` (+ `tensorflow`) only for `prepare_dataset.py`.

```bash
pip install torch rdkit pandas scikit-learn matplotlib
git clone https://github.com/JinYSun/D-GCAN.git vendor/D-GCAN   # upstream model code
```

- Local scripts (`prepare_dataset.py`, `v2`, `v3`, `v4`, `v6`) use paths
  relative to the repo and import D-GCAN from `vendor/D-GCAN/DGCAN`. On a
  machine without CUDA, set `preprocess.device = torch.device('cpu')` after
  import (upstream hard-codes `cuda`).
- GPU scripts (`v5`, `v7`, `v8`, `v9`) are written for a Colab VM (`/content/...`
  paths) and clone the upstream repo automatically. For `v9`, set
  `DATASET_OVERRIDE = "BACE"` (or `"ClinTox"` / `"Tox21"`) in the kernel first.
- `v1` (and the earlier drafts `train_dataset.py`, `train_bbbp.py`) are the
  original Colab notebooks' code, kept unchanged for reference; they expect the
  project on a mounted Google Drive at `/content/drive/MyDrive/DGCAN_Project`.

A full 5-seed run takes ~34 min (BBBP), ~41 min (BACE) and ~30 min (ClinTox)
on a T4 GPU.

---

## Roadmap

- [x] Diagnose and fix the D-GCAN training bug
- [x] Multi-seed BBBP baseline
- [x] BACE, ClinTox multi-seed baselines
- [ ] Tox21 (SR-MMP) multi-seed baseline
- [ ] Random vs. scaffold split comparison
- [ ] Architecture comparison (GCN / GAT / GraphSAGE / D-GCAN)
- [ ] Ablations (GCN, GAT, fingerprint and attention components)
- [ ] Explainability (GNNExplainer / attention case studies)

---

## Acknowledgements

The model architecture and original training code are from **D-GCAN** by
Jinyu Sun et al. — https://github.com/JinYSun/D-GCAN (BSD 3-Clause License).
This repository does not redistribute that code; it is cloned at run time.
Datasets are from [MoleculeNet](https://moleculenet.org/).
