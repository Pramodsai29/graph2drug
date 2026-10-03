"""
Reusable dataset preparation pipeline: raw CSV -> D-GCAN-compatible format
-> scaffold split. Mirrors the cleaning steps originally used for BBBP:
select smiles+label columns -> drop missing -> drop duplicate molecules
(by canonical SMILES) -> drop invalid/multi-fragment SMILES -> write
whitespace-separated {smiles}\t{label} files, no header line (matches
vendor/D-GCAN/DGCAN/preprocess.py's create_dataset, which always discards
the file's first line as if it were a header).

Usage: python prepare_dataset.py <DATASET_NAME>
DATASET_NAME must be a key in DATASET_SPECS below.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

DATASET_SPECS = {
    "BACE": {"raw_file": "bace.csv", "smiles_col": "mol", "label_col": "Class"},
    "ClinTox": {"raw_file": "clintox.csv", "smiles_col": "smiles", "label_col": "CT_TOX"},
    "Tox21": {"raw_file": "tox21.csv", "smiles_col": "smiles", "label_col": "SR-MMP"},
}


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def prepare(dataset_name):
    spec = DATASET_SPECS[dataset_name]
    raw_path = RAW_DIR / spec["raw_file"]

    print("=" * 70)
    print(f"Preparing {dataset_name} from {raw_path.name}")
    print("=" * 70)

    df = pd.read_csv(raw_path)
    n_raw = len(df)
    print(f"Raw rows: {n_raw}")

    df = df[[spec["smiles_col"], spec["label_col"]]].copy()
    df.columns = ["smiles", "label"]

    df = df.dropna(subset=["smiles", "label"])
    n_after_dropna = len(df)
    print(f"After dropping missing smiles/label: {n_after_dropna}")

    df["label"] = df["label"].astype(int)

    # Exclude multi-fragment SMILES (contain '.') — preprocess.py filters these
    # at load time anyway; filtering here keeps our reported counts accurate.
    df = df[~df["smiles"].str.contains(r"\.", regex=True)]
    n_after_fragment_filter = len(df)
    print(f"After removing multi-fragment SMILES: {n_after_fragment_filter}")

    df["canonical_smiles"] = df["smiles"].apply(canonicalize)
    n_invalid = df["canonical_smiles"].isna().sum()
    df = df.dropna(subset=["canonical_smiles"])
    print(f"Removed {n_invalid} RDKit-invalid SMILES -> {len(df)} remaining")

    df = df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    n_after_dedup = len(df)
    print(f"After removing duplicate molecules (by canonical structure): {n_after_dedup}")

    pos = int(df["label"].sum())
    neg = len(df) - pos
    print(f"Class distribution: neg={neg} ({100*neg/len(df):.2f}%) pos={pos} ({100*pos/len(df):.2f}%)")

    full_path = PROCESSED_DIR / f"{dataset_name}.txt"
    with open(full_path, "w") as f:
        for s, l in zip(df["canonical_smiles"], df["label"]):
            f.write(f"{s}\t{l}\n")
    print(f"Wrote cleaned dataset: {full_path} ({len(df)} molecules)")

    # ==========================================================
    # Scaffold split (DeepChem ScaffoldSplitter, 80/10/10, seed=42)
    # — same method/seed used for BBBP.
    # ==========================================================

    X = np.zeros(len(df))
    dc_dataset = dc.data.NumpyDataset(X=X, y=df["label"].values, ids=df["canonical_smiles"].values)

    splitter = dc.splits.ScaffoldSplitter()
    train, valid, test = splitter.train_valid_test_split(
        dc_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=SEED
    )

    for split_name, split_data in [("train", train), ("valid", valid), ("test", test)]:
        out_path = PROCESSED_DIR / f"{dataset_name}_{split_name}.txt"
        with open(out_path, "w") as f:
            for smi, label in zip(split_data.ids, split_data.y):
                f.write(f"{smi}\t{int(label)}\n")
        print(f"{split_name}: {len(split_data.ids)} molecules -> {out_path}")

    print(f"\n{dataset_name} preparation complete.\n")
    return {
        "dataset": dataset_name,
        "raw_rows": n_raw,
        "after_dropna": n_after_dropna,
        "after_fragment_filter": n_after_fragment_filter,
        "invalid_smiles_removed": int(n_invalid),
        "after_dedup": n_after_dedup,
        "neg": neg,
        "pos": pos,
        "train": len(train.ids),
        "valid": len(valid.ids),
        "test": len(test.ids),
    }


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(DATASET_SPECS.keys())
    summaries = [prepare(name) for name in names]

    summary_df = pd.DataFrame(summaries)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(PROCESSED_DIR / "dataset_preparation_summary.csv", index=False)
