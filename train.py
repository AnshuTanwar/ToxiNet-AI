
import argparse
import os
import pickle
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

# ── Constants ────────────────────────────────────────────────────────────────

ENDPOINTS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

# Human-readable biological pathway for each endpoint (used in UI narrative)
ENDPOINT_BIO = {
    "NR-AR":        ("Androgen Receptor",         "Endocrine disruption, hormonal cancers"),
    "NR-AR-LBD":    ("Androgen Receptor LBD",     "Ligand-binding domain — male hormone pathway"),
    "NR-AhR":       ("Aryl Hydrocarbon Receptor",  "Dioxin-like toxicity, immune suppression"),
    "NR-Aromatase": ("Aromatase Enzyme",           "Estrogen biosynthesis disruption"),
    "NR-ER":        ("Estrogen Receptor",          "Endocrine disruption, breast cancer risk"),
    "NR-ER-LBD":    ("Estrogen Receptor LBD",     "Estrogen ligand binding — reproductive toxicity"),
    "NR-PPAR-gamma":("PPAR-gamma Receptor",       "Metabolic disruption, obesity/diabetes pathway"),
    "SR-ARE":       ("Antioxidant Response",       "Oxidative stress, reactive metabolite formation"),
    "SR-ATAD5":     ("DNA Damage Response",        "Genotoxicity, potential carcinogenicity"),
    "SR-HSE":       ("Heat Shock Response",        "Cellular stress, protein misfolding"),
    "SR-MMP":       ("Mitochondrial Membrane",     "Mitochondrial toxicity, cell death"),
    "SR-p53":       ("p53 Tumour Suppressor",      "Genotoxic stress, DNA damage signalling"),
}

MORGAN_RADIUS = 2
MORGAN_BITS   = 2048
MODELS_DIR    = "models"

# ── Feature engineering ──────────────────────────────────────────────────────

def smiles_to_features(smiles: str):
    """
    Convert a SMILES string to a feature vector.
    Returns (feature_array, feature_names) or (None, None) if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Morgan fingerprint (circular, 2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    fp_arr = np.array(fp)
    fp_names = [f"morgan_{i}" for i in range(MORGAN_BITS)]

    # RDKit molecular descriptors (200 physicochemical properties)
    desc_names = [d[0] for d in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    desc_vals = np.array(calc.CalcDescriptors(mol), dtype=float)

    # Combine
    features = np.concatenate([fp_arr, desc_vals])
    names    = fp_names + desc_names
    return features, names


def build_feature_matrix(df: pd.DataFrame):
    """Build feature matrix for all valid molecules."""
    print("Generating molecular features...")
    features, valid_idx, feat_names = [], [], None

    for i, row in df.iterrows():
        feat, names = smiles_to_features(row["smiles"])
        if feat is not None:
            # Replace inf descriptors with NaN, letting XGBoost handle NaNs naturally
            feat[np.isinf(feat)] = np.nan
            feat = np.clip(feat, -1e6, 1e6)
            features.append(feat)
            valid_idx.append(i)
            if feat_names is None:
                feat_names = names

    X = np.array(features, dtype=np.float32)
    print(f"  Valid molecules: {len(valid_idx)} / {len(df)}")
    print(f"  Feature dimensions: {X.shape[1]}")
    return X, valid_idx, feat_names


# ── Model training ────────────────────────────────────────────────────────────

def train_endpoint(X, y_raw, endpoint_name):
    """
    Train one XGBoost model for a single endpoint.
    Handles missing labels and class imbalance.
    """
    # Only use rows that have a label for this endpoint
    labeled_mask = y_raw != -1
    X_lab = X[labeled_mask]
    y_lab = y_raw[labeled_mask]

    if len(y_lab) < 50 or y_lab.sum() < 10:
        print(f"  [{endpoint_name}] Skipped — too few samples")
        return None, None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_lab, y_lab, test_size=0.2, random_state=42, stratify=y_lab
    )

    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print(f"  [{endpoint_name}] ROC-AUC = {auc:.4f}  (test n={len(y_te)})")
    return model, auc


# ── Main ─────────────────────────────────────────────────────────────────────

def main(data_path: str):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} molecules from {data_path}")

    # Build features
    X, valid_idx, feat_names = build_feature_matrix(df)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    # Save feature names for SHAP later
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feat_names, f)

    # Save valid SMILES mapping
    df_valid[["mol_id", "smiles"]].to_csv(
        os.path.join(MODELS_DIR, "valid_molecules.csv"), index=False
    )

    # Train one model per endpoint
    print("\nTraining multi-task models...")
    results = {}
    for ep in ENDPOINTS:
        if ep not in df_valid.columns:
            continue
        # Convert labels: "" → -1 (missing), "0" → 0, "1" → 1
        y_raw = df_valid[ep].apply(
            lambda v: -1 if (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "" else int(float(v))
        ).values

        model, auc = train_endpoint(X, y_raw, ep)
        if model is not None:
            results[ep] = {"model": model, "auc": auc}

    # Save all models together
    payload = {
        "models":      {ep: r["model"] for ep, r in results.items()},
        "aucs":        {ep: r["auc"]   for ep, r in results.items()},
        "feat_names":  feat_names,
        "endpoint_bio": ENDPOINT_BIO,
        "endpoints":   ENDPOINTS,
    }
    model_path = os.path.join(MODELS_DIR, "safedrug_models.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nAll models saved to {model_path}")
    print("\nSummary:")
    print(f"  {'Endpoint':<16} {'ROC-AUC':>8}")
    print(f"  {'-'*26}")
    for ep, r in results.items():
        print(f"  {ep:<16} {r['auc']:>8.4f}")
    avg = np.mean([r["auc"] for r in results.values()])
    print(f"  {'AVERAGE':<16} {avg:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="tox21.csv", help="Path to tox21.csv")
    args = parser.parse_args()
    main(args.data)
