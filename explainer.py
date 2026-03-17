"""
SafeDrug AI — Phase 3
SHAP explainability engine.
This module is imported by the Streamlit app.
It handles: prediction, SHAP explanation, atom highlighting, modification suggestions.
"""

import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import shap
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

MORGAN_RADIUS = 2
MORGAN_BITS   = 2048

# Risk level thresholds
def risk_level(prob):
    if prob >= 0.7:  return "High",   "#E24B4A"
    if prob >= 0.4:  return "Medium", "#EF9F27"
    return "Low", "#1D9E75"


def load_models(model_path="models/safedrug_models.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    fp_arr = np.array(fp, dtype=np.float32)

    desc_names = [d[0] for d in Descriptors.descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    desc_vals = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
    desc_vals = np.nan_to_num(desc_vals, nan=0.0, posinf=0.0, neginf=0.0)

    features = np.concatenate([fp_arr, desc_vals]).reshape(1, -1)
    feat_names = [f"morgan_{i}" for i in range(MORGAN_BITS)] + desc_names
    return features, feat_names, mol


def predict_all_endpoints(smiles, payload):
    """
    Returns dict: endpoint → {prob, risk_label, risk_color, shap_top, auc}
    """
    features, feat_names, mol = smiles_to_features(smiles)
    if features is None:
        return None

    results = {}
    for ep in payload["endpoints"]:
        model = payload["models"].get(ep)
        if model is None:
            continue

        prob = float(model.predict_proba(features)[0][1])
        label, color = risk_level(prob)

        # SHAP explanation for this endpoint
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(features)

        # shap_vals may be list (binary) or array
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]   # positive class
        else:
            sv = shap_vals[0]

        # Top 10 features by absolute SHAP value
        top_idx = np.argsort(np.abs(sv))[::-1][:10]
        top_features = [
            {
                "name":  feat_names[i],
                "shap":  float(sv[i]),
                "value": float(features[0][i]),
            }
            for i in top_idx
        ]

        results[ep] = {
            "prob":        prob,
            "risk_label":  label,
            "risk_color":  color,
            "shap_top":    top_features,
            "auc":         payload["aucs"].get(ep, 0.0),
            "bio":         payload["endpoint_bio"].get(ep, ("Unknown", "")),
        }

    return results, mol


def get_atom_shap_weights(smiles, endpoint, payload):
    """
    Map Morgan fingerprint SHAP contributions back to atoms.
    Returns list of (atom_idx, weight) for heatmap colouring.
    """
    features, feat_names, mol = smiles_to_features(smiles)
    if features is None or mol is None:
        return []

    model = payload["models"].get(endpoint)
    if model is None:
        return []

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(features)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]

    # Map each Morgan bit to the atoms that set it
    bi = {}
    AllChem.GetMorganFingerprintAsBitVect(
        mol, MORGAN_RADIUS, nBits=MORGAN_BITS, bitInfo=bi
    )

    atom_weights = np.zeros(mol.GetNumAtoms())
    for bit_idx, atom_envs in bi.items():
        bit_shap = sv[bit_idx]   # SHAP contribution of this fingerprint bit
        for atom_idx, radius in atom_envs:
            atom_weights[atom_idx] += bit_shap

    # Normalise to [-1, 1]
    max_abs = np.max(np.abs(atom_weights)) + 1e-9
    atom_weights = atom_weights / max_abs

    return [(i, float(w)) for i, w in enumerate(atom_weights)]


def render_molecule_heatmap(smiles, endpoint, payload, width=500, height=400):
    """
    Render molecule image with atom-level SHAP heatmap.
    Red atoms = toxic contributors, Green = safe contributors.
    Returns PIL Image using RDKit's built-in PNG renderer.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_weights = get_atom_shap_weights(smiles, endpoint, payload)
    weight_map = dict(atom_weights)

    # Build colour dict: positive SHAP = red, negative = green
    atom_cols = {}
    for atom_idx, w in weight_map.items():
        intensity = min(abs(w), 1.0)
        if w > 0:
            atom_cols[atom_idx] = (1.0, 1.0 - intensity * 0.8, 1.0 - intensity * 0.8)
        else:
            atom_cols[atom_idx] = (1.0 - intensity * 0.8, 1.0, 1.0 - intensity * 0.8)

    highlight_atoms = list(weight_map.keys())

    # Use RDKit's direct PNG renderer — no system libraries needed
    img = Draw.MolToImage(
        mol,
        size=(width, height),
        highlightAtoms=highlight_atoms,
        highlightColor=None,
        highlightMap=atom_cols,
    )
    return img


def suggest_modifications(smiles, results, payload):
    """
    SHAP-guided modification suggestions.
    Identifies the molecular descriptor with the highest positive SHAP
    contribution to the highest-risk endpoint and suggests a structural change.
    """
    if not results:
        return []

    suggestions = []

    # Find top 3 risky endpoints
    ranked = sorted(results.items(), key=lambda x: x[1]["prob"], reverse=True)[:3]

    for ep, data in ranked:
        if data["prob"] < 0.4:
            continue
        pathway, mechanism = data["bio"]

        # Top SHAP feature driving toxicity
        toxic_features = [f for f in data["shap_top"] if f["shap"] > 0]
        if not toxic_features:
            continue

        top_feat = toxic_features[0]
        feat_name = top_feat["name"]
        shap_contribution = top_feat["shap"]

        # Map feature name to a human-readable modification hint
        hint = _feature_to_modification_hint(feat_name, smiles)

        suggestions.append({
            "endpoint":    ep,
            "pathway":     pathway,
            "risk":        data["prob"],
            "feature":     feat_name,
            "shap":        shap_contribution,
            "hint":        hint,
            "new_risk_est": max(0.0, data["prob"] - shap_contribution * 0.5),
        })

    return suggestions


def _feature_to_modification_hint(feat_name, smiles):
    """Map descriptor/fingerprint names to actionable chemistry suggestions."""
    mol = Chem.MolFromSmiles(smiles)

    # Substructure-based hints (check actual molecule)
    substructure_hints = {
        "c1ccc([N+](=O)[O-])cc1": "Remove or replace nitro (-NO₂) group — known genotoxic substructure",
        "C(=O)Cl":                "Replace acyl chloride with ester — reduces electrophilicity",
        "S(=O)(=O)":              "Consider sulfonamide bioisostere to reduce SR-ARE activation",
        "[N;H2]c1ccccc1":         "Aromatic amine oxidation risk — consider N-methylation or removal",
    }
    for smarts, hint in substructure_hints.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol and mol.HasSubstructMatch(pattern):
            return hint

    # Descriptor-based hints
    desc_hints = {
        "MolLogP":    "Reduce lipophilicity (logP) — consider adding polar groups to improve ADMET",
        "TPSA":       "Increase TPSA by adding H-bond donors/acceptors — improves membrane selectivity",
        "NumHAcceptors": "Balance H-bond acceptors to reduce non-specific binding",
        "RingCount":  "Reduce ring count — fewer aromatic rings lower intercalation risk",
        "NumRotatableBonds": "Reduce rotatable bonds to improve metabolic stability",
        "MaxPartialCharge": "Reduce maximum partial charge — lowers reactive site activity",
        "MinPartialCharge": "Balance charge distribution to reduce electrophilic reactivity",
    }
    for desc, hint in desc_hints.items():
        if desc in feat_name:
            return hint

    # Generic fingerprint hint
    if feat_name.startswith("morgan_"):
        return "Modify the molecular substructure associated with this fingerprint bit — consider scaffold hopping"

    return "Structural optimisation suggested for this molecular region"

def lipinski_rules(smiles):
    """
    Check Lipinski's Rule of Five for drug-likeness.
    Returns dict with each rule, value, pass/fail, and overall verdict.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    from rdkit.Chem import Descriptors as D
    mw      = D.MolWt(mol)
    logp    = D.MolLogP(mol)
    hdonors = D.NumHDonors(mol)
    haccept = D.NumHAcceptors(mol)
    tpsa    = D.TPSA(mol)
    rotbonds= D.NumRotatableBonds(mol)

    rules = [
        {
            "rule":    "Molecular weight",
            "value":   round(mw, 1),
            "unit":    "Da",
            "limit":   "≤ 500 Da",
            "pass":    mw <= 500,
            "detail":  "Large molecules have poor oral absorption"
        },
        {
            "rule":    "LogP (lipophilicity)",
            "value":   round(logp, 2),
            "unit":    "",
            "limit":   "≤ 5",
            "pass":    logp <= 5,
            "detail":  "High logP = poor solubility, membrane toxicity risk"
        },
        {
            "rule":    "H-bond donors",
            "value":   hdonors,
            "unit":    "",
            "limit":   "≤ 5",
            "pass":    hdonors <= 5,
            "detail":  "Too many donors = poor membrane permeability"
        },
        {
            "rule":    "H-bond acceptors",
            "value":   haccept,
            "unit":    "",
            "limit":   "≤ 10",
            "pass":    haccept <= 10,
            "detail":  "Too many acceptors = poor oral bioavailability"
        },
        {
            "rule":    "TPSA",
            "value":   round(tpsa, 1),
            "unit":    "Ų",
            "limit":   "≤ 140 Ų",
            "pass":    tpsa <= 140,
            "detail":  "High TPSA = poor GI absorption and CNS penetration"
        },
        {
            "rule":    "Rotatable bonds",
            "value":   rotbonds,
            "unit":    "",
            "limit":   "≤ 10",
            "pass":    rotbonds <= 10,
            "detail":  "Excess flexibility reduces oral bioavailability"
        },
    ]

    core_passes = sum(1 for r in rules[:4] if r["pass"])  # Ro5 = first 4
    verdict = (
        "Excellent"  if core_passes == 4 else
        "Acceptable" if core_passes == 3 else
        "Poor"       if core_passes == 2 else
        "Rejected"
    )
    return {"rules": rules, "verdict": verdict, "core_passes": core_passes}