# Graph2Drug — Review Demo Guide

Total time: ~8 minutes. Everything runs on a laptop CPU; nothing needs a GPU or internet.

## Before the review (once)

```bash
cd ~/Documents/graph2drug
source .venv/bin/activate          # Python env with torch, rdkit, scikit-learn, pandas, matplotlib
python experiments/demo_predict.py --dataset BBBP --verify    # warm-up run, ~10 s
```

If setting up on a new machine:

```bash
git clone https://github.com/Pramodsai29/graph2drug.git && cd graph2drug
git clone https://github.com/JinYSun/D-GCAN.git vendor/D-GCAN
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Web demo (recommended for the review)

Double-click **`start_demo.command`** in the `graph2drug` folder (or run
`streamlit run app.py`). The app opens at http://localhost:8501 in your browser.
Wait ~15 s on first load while the trained model loads. Keep the Terminal window open
during the demo; close it to stop the app.

| Tab | What to show |
|---|---|
| 🔬 Predict | Pick BBBP or BACE, choose an example molecule — the structure is drawn and the model's probability and verdict appear. Type any SMILES a reviewer suggests. |
| ⚙️ How it works | Architecture diagram + the bug (sigmoid then cross-entropy) next to our fix. |
| 📊 Results | Original 0.466 vs fixed scores, AUC chart, and the training curve where the original flat-lines at 0.5. |
| ✅ Model check | Click **Run check** — re-scores the whole test set live and matches the reported AUC exactly. |

Suggested order: Results (the problem and the payoff) → How it works (the fix) →
Predict (live) → Model check (proof). The terminal commands below are a backup.

## 1. The problem (2 min)

Open these images from `results/`:

- `BBBP_v2_score_histogram.png` — the **original** D-GCAN gives every molecule a score of ~1.0.
- `BBBP_v3_training_curves.png` — validation AUC stays at 0.5 for all 140 epochs.

**Say:** "When we trained the original published D-GCAN code on BBBP, test AUC was
0.466 — worse than a coin flip. The model predicted *every* molecule as positive."

## 2. Finding and fixing the bug — show the code (2 min)

**Diagnosis** (`experiments/train_dataset_v3.py`): we checked logits, gradients and
input sensitivity without retraining. Gradients were 1e-10 to 1e-7, i.e. the network
had stopped learning.

**Root cause** (upstream `vendor/D-GCAN/DGCAN/DGCAN.py`): `mlp()` applies
`torch.sigmoid()` and `forward_classifier()` then passes that into
`F.cross_entropy()`, which applies softmax again. The sigmoid saturates to exactly 0/1
and the gradient dies.

**Our fix** — open `experiments/train_dataset_v8.py`, function
`fixed_forward_classifier`: raw logits go straight into `F.cross_entropy`.

**Say:** "We did not edit the original library; we replace one method on the model
object, so the fix is small, isolated and easy to verify."

## 3. Live demo — the model working (2 min)

**3a. Blood–brain barrier (BBBP):**

```bash
python experiments/demo_predict.py --dataset BBBP --smiles \
  "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
  "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3" \
  "CC1(C)SC2C(NC(=O)CC3=CC=CC=C3)C(=O)N2C1C(=O)O" \
  "OCC1OC(OC2(CO)OC(CO)C(O)C2O)C(O)C(O)C1O"
```

| Molecule | Expected | Model |
|---|---|---|
| Caffeine | crosses (it's a stimulant) | 0.97 → crosses ✓ |
| Diazepam (Valium) | crosses (acts on the brain) | 0.98 → crosses ✓ |
| Penicillin G | barely crosses | 0.07 → does not cross ✓ |
| Sucrose (table sugar) | does not cross | 0.02 → does not cross ✓ |

**3b. BACE-1 inhibition (Alzheimer's target)** — molecules from the held-out test set:

```bash
python experiments/demo_predict.py --dataset BACE --smiles \
  "Cc1cc(C2(c3cccc(C#CC4CC4)c3)N=C(N)c3c(F)cccc32)cn(C)c1=O" \
  "Nc1nc(CCc2ccc3cc[nH]c3c2)cc(=O)[nH]1"
```

First is a known inhibitor (model ≈ 0.82 → inhibits ✓); second is a known
non-inhibitor (model prints 0.0000 → does not inhibit ✓).

**3c. Prove it is the real model, not hard-coded:**

```bash
python experiments/demo_predict.py --dataset BBBP --verify
```

Re-scores all 183 test molecules from scratch and reproduces the reported test AUC
(0.6523) exactly.

You can also type any SMILES a reviewer suggests (copy from PubChem). An invalid one
prints "invalid SMILES" instead of crashing.

## 4. Results (2 min)

| | Test ROC-AUC (5 seeds, scaffold split) |
|---|---|
| Original D-GCAN, BBBP | 0.466 (broken) |
| Fixed, BBBP | **0.640 ± 0.012** |
| Fixed, BACE | **0.794 ± 0.019** |
| Fixed, ClinTox | **0.842 ± 0.029** |
| Tox21 | in progress |

Then show the GitHub repo: https://github.com/Pramodsai29/graph2drug

## Likely questions

- **Why is BBBP lower than BACE/ClinTox?** Scaffold splitting puts structurally novel
  molecules in the test set. BBBP's test molecules are measurably less similar to the
  training set (Tanimoto 0.415 vs 0.472 for validation), so it is the hardest
  generalisation test. That is why we report the honest test number, not validation.
- **Why 5 seeds?** One run can be lucky. Mean ± std over 5 seeds shows the result is
  reproducible (BBBP std is only 0.012).
- **Is the model ever wrong?** Yes — e.g. it says betaine (a charged molecule,
  `C[N+](C)(C)CC(=O)[O-]`) crosses the barrier, which is wrong. BBBP AUC 0.64 means it
  ranks correctly about 64% of the time; it is a screening aid, not an oracle.
- **What is the threshold?** A molecule is called positive if probability > 0.15,
  the threshold used throughout the original D-GCAN evaluation.
- **What's next?** Tox21, random vs scaffold split comparison, comparing GCN/GAT/
  GraphSAGE, ablations, and explainability (which atoms drive a prediction).
