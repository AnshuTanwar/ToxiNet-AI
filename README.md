# ToxiNet AI
### Multi-task Drug Toxicity Prediction with SHAP Explainability

Built for CodeCure AI Hackathon — Track A: Drug Toxicity Prediction

---

## What it does

ToxiNet AI takes a molecular SMILES string and predicts toxicity risk across all **12 Tox21 biological assay endpoints** simultaneously. Unlike single-label classifiers, it tells you *which biological pathway* is at risk and *which exact atoms* are responsible — powered by SHAP explainability.

**Key outputs:**
- Toxicity fingerprint radar chart (12 pathways at once)
- Atom-level heatmap showing which molecular regions drive toxicity
- SHAP feature importance per endpoint
- Biological pathway interpretation (what the toxicity *means* clinically)
- SHAP-guided modification suggestions

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (run once — takes ~5–10 min)
python train.py --data tox21.csv

# 3. Launch the app
streamlit run app.py
```

---

## Project structure

```
safedrug_ai/
├── train.py          # Phase 1+2: Feature engineering + model training
├── explainer.py      # Phase 3: SHAP explainability engine
├── app.py            # Phase 4: Streamlit dashboard
├── requirements.txt
├── models/           # Generated after training
│   └── safedrug_models.pkl
└── tox21.csv         # Place your dataset here
```

---

## The 12 Tox21 endpoints

| Endpoint | Biological Pathway | Clinical Relevance |
|---|---|---|
| NR-AR | Androgen Receptor | Endocrine disruption, hormonal cancers |
| NR-AR-LBD | Androgen Receptor LBD | Male hormone pathway |
| NR-AhR | Aryl Hydrocarbon Receptor | Dioxin-like toxicity, immune suppression |
| NR-Aromatase | Aromatase Enzyme | Estrogen biosynthesis disruption |
| NR-ER | Estrogen Receptor | Endocrine disruption, breast cancer risk |
| NR-ER-LBD | Estrogen Receptor LBD | Reproductive toxicity |
| NR-PPAR-gamma | PPAR-gamma Receptor | Metabolic disruption |
| SR-ARE | Antioxidant Response | Oxidative stress, reactive metabolites |
| SR-ATAD5 | DNA Damage Response | Genotoxicity, carcinogenicity |
| SR-HSE | Heat Shock Response | Cellular stress, protein misfolding |
| SR-MMP | Mitochondrial Membrane | Mitochondrial toxicity, cell death |
| SR-p53 | p53 Tumour Suppressor | DNA damage signalling |

---

## Technical approach

- **Feature engineering:** Morgan fingerprints (2048-bit, radius 2) + 200 RDKit molecular descriptors
- **Model:** XGBoost per endpoint, class imbalance handled with SMOTE
- **Explainability:** SHAP TreeExplainer — maps feature contributions back to atom positions
- **Dataset:** Tox21 (7,831 compounds) — the standard benchmark for ML toxicity prediction

---

## Demo molecules

| Molecule | SMILES | Expected result |
|---|---|---|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | Low risk — clean radar |
| Tamoxifen | `CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1` | High NR-ER risk |
| Nitrobenzene | `O=[N+]([O-])c1ccccc1` | High SR-ARE, SHAP highlights NO₂ group |
| Caffeine | `Cn1cnc2c1c(=O)n(C)c(=O)n2C` | Low risk |
| Bisphenol A | `CC(c1ccc(O)cc1)(c1ccc(O)cc1)C` | NR-ER / NR-AR endocrine risk |
