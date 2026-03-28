# ToxiNet AI

> **Multi-task drug toxicity prediction · SHAP explainability · Biological pathway analysis · ZINC250k enrichment**

Built for **CodeCure AI Hackathon — Track A: Drug Toxicity Prediction**

---

## The Problem We're Solving

90% of drug candidates fail in clinical trials. The single biggest reason is **unexpected toxicity** — discovered only after years of development and hundreds of crores have already been spent. Existing tools tell you *whether* a molecule is toxic. They cannot tell you *which biological pathway* is being disrupted, *which atoms* are responsible, or *how to fix it*.

**ToxiNet AI answers all three questions simultaneously.**

---

## What ToxiNet AI Does

ToxiNet AI takes any molecular SMILES string and delivers a complete toxicity intelligence report across **12 Tox21 biological assay endpoints** — not a single binary label, but 12 simultaneous pathway-specific risk scores, each with atom-level chemical explanations grounded in experimental evidence from real assay data.

### The scientific insight that sets us apart

When Tamoxifen enters our system, it scores **0.99 on NR-Aromatase** (the enzyme controlling estrogen biosynthesis). We did not tell the model this. It learned Tamoxifen's known pharmacological mechanism from molecular structure alone. That is scientific validation — not just a benchmark number.

When Bisphenol A enters our system, it **passes all Lipinski drug-likeness rules** — a traditional screening tool would approve it. Our toxicity models flag it for estrogen receptor (NR-ER) disruption. That is why BPA was eventually banned from food containers worldwide, a fact that took decades to discover using traditional methods.

---

## Feature Overview

| # | Feature | What it does |
|---|---------|-------------|
| 1 | **Multi-task toxicity prediction** | 12 XGBoost models run simultaneously — one per Tox21 biological assay — each returning an independent risk probability |
| 2 | **Tanimoto similarity search** | Finds the top 5 most structurally similar molecules in the Tox21 training data using Morgan fingerprint Tanimoto coefficients. Shows their confirmed experimental toxicity labels, grounding predictions in real assay evidence |
| 3 | **Toxicity fingerprint radar** | 12-axis spider chart showing the complete biological risk profile of a molecule at a glance. Dashed threshold rings mark medium (0.4) and high (0.7) risk levels |
| 4 | **Extended drug profile (QED · SAS · ADMET)** | Quantitative drug-likeness score (QED), synthetic accessibility score (SAS from ZINC250k), GI absorption, BBB penetration estimate, bioavailability (Veber rules), mutagenicity alerts, PAINS substructure flags, CYP inhibition risks, and Murcko scaffold |
| 5 | **Atom-level SHAP heatmap** | Maps SHAP TreeExplainer values back to individual atoms using Morgan fingerprint bitInfo. Red atoms increase toxicity, green atoms reduce it — the exact chemical explanation per endpoint |
| 6 | **Lipinski Rule of Five** | Six physicochemical properties (MW, logP, H-donors, H-acceptors, TPSA, rotatable bonds) evaluated with Excellent / Acceptable / Poor verdict |
| 7 | **SHAP feature contributions** | Horizontal bar chart of top 10 molecular features driving each of the 12 endpoint predictions — red bars increase risk, green bars reduce it |
| 8 | **SHAP-guided modification suggestions** | Matches highest SHAP contributors to known toxic substructures (nitro groups, acyl chlorides, aromatic amines) and suggests specific structural changes with estimated post-modification risk scores |
| 9 | **Molecule comparison mode** | Two SMILES inputs, overlaid radar charts in red vs blue, side-by-side endpoint tables, a risk delta table sorted by biggest risk reduction, and Lipinski comparison for both molecules |
| 10 | **Batch screening** | Upload any CSV with a `smiles` column. Screens all molecules with a live progress bar, ranks by average toxicity score, shows risk distribution chart, exports full results as CSV |
| 11 | **PDF report export** | Dark-themed A4 report with molecule image, radar chart, 12-endpoint table, Lipinski panel, SHAP chart, and modification suggestions — formatted like a professional drug discovery deliverable |
| 12 | **Molecule sketcher (JSME)** | Embedded JSME molecule drawing tool — draw atoms and bonds visually, extract SMILES with one click, paste directly into analysis tab |

---

## Model Performance

| Endpoint | Biological Pathway | ROC-AUC |
|----------|-------------------|---------|
| NR-AR | Androgen Receptor | 0.717 |
| NR-AR-LBD | Androgen Receptor LBD | 0.899 |
| NR-AhR | Aryl Hydrocarbon Receptor | 0.911 |
| NR-Aromatase | Aromatase Enzyme | 0.863 |
| NR-ER | Estrogen Receptor | 0.733 |
| NR-ER-LBD | Estrogen Receptor LBD | 0.836 |
| NR-PPAR-gamma | PPAR-gamma Receptor | 0.843 |
| SR-ARE | Antioxidant Response Element | 0.861 |
| SR-ATAD5 | DNA Damage Response | 0.898 |
| SR-HSE | Heat Shock Response Element | 0.849 |
| SR-MMP | Mitochondrial Membrane Potential | **0.932** |
| SR-p53 | p53 Tumour Suppressor | 0.895 |
| **Average** | | **0.853** |

NR-AR and NR-ER score lower because they have the most structurally complex receptor-binding mechanisms in the Tox21 benchmark — these two endpoints consistently score lowest across all published models in the literature. Our 0.717 and 0.733 are consistent with state-of-the-art results.

---

## Datasets Used

### Primary — Tox21 (7,823 valid molecules)
The standard benchmark for AI toxicity prediction. Contains ~12,000 chemical compounds with toxicity assay results across 12 nuclear receptor and stress response pathways. Used for training all 12 XGBoost models.

### Secondary — ZINC250k (249,455 drug-like molecules)
A curated subset of the ZINC compound database. Contains SMILES strings with precomputed logP, QED, and SAS properties. Used in two ways:

1. **Property enrichment** — each Tox21 training molecule is annotated with its ZINC250k QED, SAS, and logP values where available, enriching the similarity search results with drug-likeness context
2. **Extended drug profiling** — QED and SAS are computed for any user-submitted molecule and displayed alongside the Lipinski assessment

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the toxicity models (run once — ~5–10 minutes)

```bash
python train.py --data tox21.csv
```

This trains 12 XGBoost models on Tox21, one per biological assay endpoint. Models are saved to `models/safedrug_models.pkl`. You will see ROC-AUC scores printed for each endpoint as training completes.

### 3. Build the similarity index (run once — ~3 minutes)

Without ZINC250k:
```bash
python build_index.py --tox21 tox21.csv
```

With ZINC250k enrichment (recommended):
```bash
python build_index.py --tox21 tox21.csv --zinc 250k_rndm_zinc_drugs_clean_3.csv
```

This builds a Tanimoto fingerprint index over all 7,823 training molecules and saves it to `models/similarity_index.pkl`. After this, every prediction is backed by experimental evidence from the closest training compounds.

### 4. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
toxinet_ai/
├── train.py                          # Feature engineering + XGBoost model training
├── explainer.py                      # SHAP explainability engine + atom heatmap
├── molecular_intelligence.py         # Tanimoto similarity search + QED/SAS/ADMET
├── build_index.py                    # One-time similarity index builder
├── reporter.py                       # PDF report generator (ReportLab)
├── app.py                            # Streamlit dashboard — all 4 tabs
├── requirements.txt
├── packages.txt                      # System dependencies for deployment
├── models/
│   ├── safedrug_models.pkl           # 12 trained XGBoost models (generated)
│   ├── similarity_index.pkl          # Tanimoto search index (generated)
│   └── feature_names.pkl             # Feature name mapping (generated)
├── tox21.csv                         # Primary dataset
└── 250k_rndm_zinc_drugs_clean_3.csv  # Secondary dataset (ZINC250k)
```

---

## Technical Approach

### Feature Engineering
Every SMILES string is converted into a 2,265-dimensional feature vector:
- **2,048 Morgan fingerprint bits** (circular substructures, radius 2) — encodes which molecular substructures are present
- **217 RDKit molecular descriptors** — physicochemical properties (MW, logP, TPSA, partial charges, ring counts, etc.)

### Multi-task Modelling
One XGBoost classifier per Tox21 endpoint (12 total). Each model is independent, allowing it to specialise on the biological mechanism of its assay. Class imbalance is addressed with `scale_pos_weight` (negative-to-positive ratio). All 12 models run in parallel at prediction time.

### SHAP Explainability
`shap.TreeExplainer` decomposes each prediction into individual feature contributions. For atom-level heatmaps, fingerprint SHAP values are mapped back to atoms using RDKit's `bitInfo` parameter — tracking which atoms contributed to each Morgan fingerprint bit, then summing their SHAP contributions. This produces genuine atom-level chemical explanations, not post-hoc approximations.

### Tanimoto Similarity Search
`DataStructs.BulkTanimotoSimilarity` runs instant bulk comparison of the query fingerprint against all 7,823 indexed training molecule fingerprints. The top 5 hits are returned with their Tox21 experimental labels, ZINC250k-enriched QED/SAS/logP values, and a cross-reference against the model's predictions — showing which predictions are supported by experimental evidence and which are model extrapolations.

### Extended Drug Profiling (ZINC250k Integration)
QED (Quantitative Estimate of Drug-likeness) combines 8 physicochemical properties into a single 0→1 score. SAS (Synthetic Accessibility Score) estimates how difficult a molecule is to synthesise on a scale of 1 (easy) to 10 (hard). Both are sourced from ZINC250k precomputed values where available, falling back to RDKit computation. ADMET properties (GI absorption, BBB penetration, bioavailability) are computed using validated rule-based methods (Lipinski, Veber).

---

## Why This Approach Wins

**Most teams will build a single binary classifier.** That collapses 12 fundamentally different biological mechanisms into one number. ToxiNet AI trains 12 independent models and tells you exactly which pathway is at risk — NR-AR (hormonal cancers), SR-MMP (mitochondrial death), SR-p53 (DNA damage) — each with its own atom-level explanation.

**Most teams will show a confusion matrix.** ToxiNet AI shows experimental evidence: "This molecule is 84% similar to 3 Tox21 training compounds confirmed toxic for SR-ARE. The prediction is backed by real assay data."

**Most teams will not use ZINC250k at all.** We use it for property enrichment across 249,455 drug-like molecules — making every similarity hit richer with precomputed QED, SAS, and logP context.

---

## Demo Molecules

| Molecule | SMILES | What to look for |
|----------|--------|-----------------|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | Clean radar, all green. QED ~0.61, SAS ~2.1 (easy). No structural alerts. Similarity hits: safe compounds. |
| Nitrobenzene | `O=[N+]([O-])c1ccccc1` | SR-ARE spike. Atom heatmap lights NO₂ red. Mutagenicity alert fires. PAINS alert fires. Similarity: confirmed SR-ARE toxic compounds. |
| Tamoxifen | `CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1` | NR-Aromatase **0.99**. CYP3A4 alert. logP 6.0 fails Lipinski. Known pharmacology confirmed from structure alone. |
| Bisphenol A | `CC(c1ccc(O)cc1)(c1ccc(O)cc1)C` | Passes ALL Lipinski rules (QED looks acceptable). But NR-ER flags endocrine disruption. This is the gap ToxiNet fills. |
| Caffeine | `Cn1cnc2c1c(=O)n(C)c(=O)n2C` | Low risk across all pathways. Clean profile. Good control showing the model does not over-predict. |

---

## Deliverables (Track A Checklist)

| Requirement | Status | Where |
|------------|--------|-------|
| GitHub Repository | ✅ | This repo |
| ML model predicting drug toxicity | ✅ | `train.py` → `models/safedrug_models.pkl` |
| Feature importance analysis | ✅ | `explainer.py` → SHAP tab in UI |
| Key molecular descriptors identified | ✅ | SHAP bar charts per endpoint |
| Visualisations of molecular properties vs toxicity | ✅ | Radar chart, atom heatmap, SHAP bars |
| Simple prediction interface | ✅ | `app.py` — 4-tab Streamlit dashboard |
| Secondary dataset (ZINC250k) used | ✅ | `molecular_intelligence.py` |

---

## Requirements

```
rdkit
xgboost
shap
imbalanced-learn
scikit-learn
pandas
numpy
matplotlib
streamlit>=1.35.0
plotly
Pillow
reportlab
kaleido
```

System packages (for deployment):
```
libxrender1
libxext6
libsm6
libgl1
```

---

*ToxiNet AI — translating molecular structure into biological insight.*
