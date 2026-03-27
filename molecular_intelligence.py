"""
ToxiNet AI — Molecular Intelligence Module
Feature 2: Tanimoto Similarity Search (Tox21 + optional ZINC250k enrichment)
Feature 4: QED, SAS, and extended ADMET properties

Build the index once:
    python build_index.py --tox21 tox21.csv
    python build_index.py --tox21 tox21.csv --zinc 250k_rndm_zinc_drugs_clean_3.csv
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

MORGAN_RADIUS = 2
MORGAN_BITS   = 2048
INDEX_PATH    = "models/similarity_index.pkl"

ENDPOINTS = [
    "NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase",
    "NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53",
]

# ═══════════════════════════════════════════════════════════
# FEATURE 4 — QED, SAS, ADMET
# ═══════════════════════════════════════════════════════════

def compute_extended_properties(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw      = Descriptors.MolWt(mol)
    logp    = Descriptors.MolLogP(mol)
    hdonors = Descriptors.NumHDonors(mol)
    haccept = Descriptors.NumHAcceptors(mol)
    tpsa    = Descriptors.TPSA(mol)
    rotbonds= Descriptors.NumRotatableBonds(mol)

    # QED
    try:
        qed_score = QED.qed(mol)
    except Exception:
        qed_score = None
    qed_label = (
        "Drug-like"  if qed_score and qed_score >= 0.67 else
        "Acceptable" if qed_score and qed_score >= 0.50 else
        "Borderline" if qed_score and qed_score >= 0.25 else
        "Poor"
    )

    # SAS (simplified fragment complexity method)
    try:
        sas_score = _compute_sas(mol)
    except Exception:
        sas_score = None
    sas_label = (
        "Easy"     if sas_score and sas_score < 3.0 else
        "Moderate" if sas_score and sas_score < 6.0 else
        "Hard"
    )

    # ADMET
    veber_pass = rotbonds <= 10 and tpsa <= 140
    bbb_score  = sum([mw < 450, 1.0 <= logp <= 4.0, tpsa < 90, hdonors <= 3])
    bbb_label  = "Likely" if bbb_score >= 3 else ("Possible" if bbb_score == 2 else "Unlikely")
    gi_pass    = mw <= 500 and logp <= 5 and hdonors <= 5 and haccept <= 10 and tpsa <= 140
    gi_label   = "High" if gi_pass and tpsa < 75 else "Moderate" if gi_pass else "Low"

    # Alerts
    cyp_alerts        = _check_cyp_alerts(mol)
    pains_alerts      = _check_pains_alerts(mol)
    mutagenicity      = _check_mutagenicity_alerts(mol)

    # Complexity
    n_rings      = rdMolDescriptors.CalcNumRings(mol)
    n_aromatic   = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_stereo     = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_heavyatoms = mol.GetNumHeavyAtoms()
    fsp3         = rdMolDescriptors.CalcFractionCSP3(mol)

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = None

    return {
        "mw": round(mw, 1), "logp": round(logp, 2),
        "hdonors": hdonors, "haccept": haccept,
        "tpsa": round(tpsa, 1), "rotbonds": rotbonds,
        "qed": round(qed_score, 3) if qed_score is not None else None,
        "qed_label": qed_label,
        "sas": round(sas_score, 2) if sas_score is not None else None,
        "sas_label": sas_label,
        "gi_absorption": gi_label, "bbb_penetrant": bbb_label,
        "veber_pass": veber_pass,
        "cyp_alerts": cyp_alerts, "pains_alerts": pains_alerts,
        "mutagenicity": mutagenicity,
        "n_rings": n_rings, "n_aromatic": n_aromatic,
        "n_stereo": n_stereo, "n_heavyatoms": n_heavyatoms,
        "fsp3": round(fsp3, 2), "scaffold": scaffold,
    }


def _compute_sas(mol):
    n_atoms   = mol.GetNumHeavyAtoms()
    n_rings   = rdMolDescriptors.CalcNumRings(mol)
    n_stereo  = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_spiro   = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridged = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    ring_info = mol.GetRingInfo()
    macrocycle = any(len(r) >= 8 for r in ring_info.AtomRings())
    complexity = (
        n_rings * 0.5 + n_stereo * 0.8 +
        n_spiro * 1.5 + n_bridged * 1.0 +
        (2.0 if macrocycle else 0.0)
    )
    sas = 1.0 + (complexity / max(n_atoms, 1)) * 12
    return float(np.clip(sas, 1.0, 10.0))


def _check_cyp_alerts(mol):
    patterns = {
        "CYP3A4": ["c1ccncc1", "C(=O)N", "c1ccc(cc1)N"],
        "CYP2D6": ["c1ccc2[nH]ccc2c1", "CCN", "c1ccnc(c1)N"],
        "CYP2C9": ["c1ccc(cc1)S(=O)(=O)", "OC(=O)", "c1cc(ccc1O)"],
    }
    alerts = []
    for cyp, smarts_list in patterns.items():
        for s in smarts_list:
            try:
                p = Chem.MolFromSmarts(s)
                if p and mol.HasSubstructMatch(p):
                    alerts.append(cyp); break
            except Exception:
                pass
    return list(set(alerts))


def _check_pains_alerts(mol):
    patterns = {
        "Rhodanine": "O=C1NC(=S)S1",
        "Catechol":  "Oc1ccccc1O",
        "Quinone":   "O=C1C=CC(=O)C=C1",
        "Nitro aromatic": "O=[N+]([O-])c1ccccc1",
        "Acrylate":  "C=CC(=O)O",
        "Aldehyde":  "[CH]=O",
    }
    alerts = []
    for name, s in patterns.items():
        try:
            p = Chem.MolFromSmarts(s)
            if p and mol.HasSubstructMatch(p):
                alerts.append(name)
        except Exception:
            pass
    return alerts


def _check_mutagenicity_alerts(mol):
    patterns = {
        "Nitro group":    "[N+](=O)[O-]",
        "Aromatic amine": "Nc1ccccc1",
        "Epoxide":        "C1OC1",
        "N-nitroso":      "N-N=O",
        "Acylating agent":"C(=O)Cl",
        "Azo compound":   "N=Nc1ccccc1",
    }
    alerts = []
    for name, s in patterns.items():
        try:
            p = Chem.MolFromSmarts(s)
            if p and mol.HasSubstructMatch(p):
                alerts.append(name)
        except Exception:
            pass
    return alerts


# ═══════════════════════════════════════════════════════════
# FEATURE 2 — Tanimoto Similarity Index
# ═══════════════════════════════════════════════════════════

def build_similarity_index(tox21_path, zinc_path=None, output_path=INDEX_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ── Load Tox21 ──────────────────────────────────────────
    print("Loading Tox21 dataset...")
    df = pd.read_csv(tox21_path)
    print(f"  {len(df)} rows loaded")

    # ── Load ZINC250k for property enrichment ───────────────
    zinc_lookup = {}
    if zinc_path and os.path.exists(zinc_path):
        print("Loading ZINC250k for QED/SAS/logP enrichment...")
        zdf = pd.read_csv(zinc_path)

        # Column detection — handles the actual file: smiles, logP, qed, SAS
        smi_col  = next((c for c in zdf.columns if c.lower() in ["smiles","smile"]), None)
        logp_col = next((c for c in zdf.columns if c.lower() in ["logp","logp"]), None)
        qed_col  = next((c for c in zdf.columns if c.lower() == "qed"), None)
        sas_col  = next((c for c in zdf.columns if c.lower() == "sas"), None)

        if smi_col:
            for _, row in zdf.iterrows():
                smi = str(row[smi_col]).strip()
                zinc_lookup[smi] = {
                    "logp": float(row[logp_col]) if logp_col and pd.notna(row[logp_col]) else None,
                    "qed":  float(row[qed_col])  if qed_col  and pd.notna(row[qed_col])  else None,
                    "sas":  float(row[sas_col])  if sas_col  and pd.notna(row[sas_col])  else None,
                }
            print(f"  {len(zinc_lookup)} ZINC250k molecules loaded")

    # ── Build fingerprint index ─────────────────────────────
    print("Computing fingerprints and building index...")
    entries, fps = [], []

    for i, row in df.iterrows():
        smi = str(row.get("smiles", "")).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)

        labels = {}
        for ep in ENDPOINTS:
            val = str(row.get(ep, "")).strip()
            if val == "1":   labels[ep] = True
            elif val == "0": labels[ep] = False

        # Prefer ZINC lookup, fall back to RDKit computed
        zinc = zinc_lookup.get(smi, {})
        try:
            rdkit_qed = float(QED.qed(mol))
        except Exception:
            rdkit_qed = None
        try:
            rdkit_sas = float(_compute_sas(mol))
        except Exception:
            rdkit_sas = None
        try:
            rdkit_logp = float(Descriptors.MolLogP(mol))
        except Exception:
            rdkit_logp = None

        entries.append({
            "smiles": smi,
            "mol_id": str(row.get("mol_id", f"MOL_{i}")),
            "labels": labels,
            "qed":    zinc.get("qed")  if zinc.get("qed")  is not None else rdkit_qed,
            "sas":    zinc.get("sas")  if zinc.get("sas")  is not None else rdkit_sas,
            "logp":   zinc.get("logp") if zinc.get("logp") is not None else rdkit_logp,
            "from_zinc": bool(zinc),
        })
        fps.append(fp)

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(df)}...")

    print(f"  Index: {len(entries)} molecules")
    payload = {"entries": entries, "fps": fps}
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved → {output_path}  ({os.path.getsize(output_path)/1024/1024:.1f} MB)")
    return payload


def load_similarity_index(index_path=INDEX_PATH):
    if not os.path.exists(index_path):
        return None
    with open(index_path, "rb") as f:
        return pickle.load(f)


def tanimoto_search(smiles, index, top_k=5, min_sim=0.0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not index:
        return []
    qfp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    sims = DataStructs.BulkTanimotoSimilarity(qfp, index["fps"])
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)

    results = []
    for idx in ranked:
        if len(results) >= top_k:
            break
        sim = sims[idx]
        if sim < min_sim:
            continue
        e = index["entries"][idx]
        results.append({
            "smiles":         e["smiles"],
            "mol_id":         e["mol_id"],
            "similarity":     round(float(sim), 3),
            "similarity_pct": round(float(sim) * 100, 1),
            "labels":         e["labels"],
            "toxic_eps":      [ep for ep, v in e["labels"].items() if v is True],
            "safe_eps":       [ep for ep, v in e["labels"].items() if v is False],
            "qed":            e.get("qed"),
            "sas":            e.get("sas"),
            "logp":           e.get("logp"),
            "from_zinc":      e.get("from_zinc", False),
            "is_identical":   sim >= 0.999,
            "note":           _sim_note(sim),
        })
    return results


def _sim_note(sim):
    if sim >= 0.95: return "Nearly identical — very high confidence"
    if sim >= 0.80: return "Same scaffold, minor substituents differ"
    if sim >= 0.60: return "Shared core scaffold"
    if sim >= 0.40: return "Related chemical class"
    return "Distant structural relative"


def interpret_similarity(sim_results, toxicity_results):
    supported, conflicting, novel = [], [], []
    for ep, pred in toxicity_results.items():
        prob = pred["prob"]
        known_toxic = sum(1 for r in sim_results if ep in r["labels"] and r["labels"][ep] is True)
        known_safe  = sum(1 for r in sim_results if ep in r["labels"] and r["labels"][ep] is False)
        total = known_toxic + known_safe
        if total == 0:
            continue
        ev = known_toxic / total
        if prob >= 0.7 and ev >= 0.5:
            supported.append({"endpoint": ep, "prob": prob,
                "evidence": f"{known_toxic}/{total} similar compounds toxic",
                "confidence": "High" if ev >= 0.7 else "Moderate"})
        elif prob >= 0.7 and ev < 0.3:
            conflicting.append({"endpoint": ep, "prob": prob,
                "evidence": f"Only {known_toxic}/{total} similar compounds toxic",
                "note": "Model extrapolating beyond training data"})
        elif prob < 0.4 and ev >= 0.7:
            novel.append({"endpoint": ep, "prob": prob,
                "evidence": f"{known_toxic}/{total} similar compounds toxic",
                "note": "Low model score but similar compounds are toxic — investigate"})
    return {"supported": supported, "conflicting": conflicting, "novel": novel}