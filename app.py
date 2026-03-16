"""
SafeDrug AI — Phase 4
Streamlit dashboard. Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import io
from PIL import Image

from explainer import (
    load_models, predict_all_endpoints,
    render_molecule_heatmap, suggest_modifications, risk_level
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SafeDrug AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin: 4px;
    }
    .risk-high   { color: #E24B4A; font-weight: 700; }
    .risk-medium { color: #EF9F27; font-weight: 700; }
    .risk-low    { color: #1D9E75; font-weight: 700; }
    .endpoint-card {
        background: #1a1d27;
        border-left: 4px solid #378ADD;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        border-bottom: 1px solid #2a2d3a;
        padding-bottom: 8px;
        margin: 20px 0 12px;
    }
    .bio-tag {
        background: #1e2840;
        border: 1px solid #2a3a60;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 12px;
        color: #7eb8f7;
        display: inline-block;
        margin: 2px;
    }
    .shap-positive { color: #E24B4A; }
    .shap-negative { color: #1D9E75; }
    .suggestion-card {
        background: #1e2433;
        border: 1px solid #2a3550;
        border-radius: 8px;
        padding: 14px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Demo molecules ────────────────────────────────────────────────────────────

DEMO_MOLECULES = {
    "Aspirin (safe — control)":         "CC(=O)Oc1ccccc1C(=O)O",
    "Tamoxifen (ER toxicity)":           "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Nitrobenzene (nitro group, toxic)": "O=[N+]([O-])c1ccccc1",
    "Caffeine (low toxicity)":           "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Bisphenol A (endocrine disruptor)": "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C",
}

# ── Load models (cached) ──────────────────────────────────────────────────────

@st.cache_resource
def get_models():
    try:
        return load_models("models/safedrug_models.pkl")
    except FileNotFoundError:
        return None

# ── Radar chart ───────────────────────────────────────────────────────────────

def make_radar_chart(results):
    endpoints = list(results.keys())
    probs = [results[ep]["prob"] for ep in endpoints]
    short_names = [ep.replace("NR-", "").replace("SR-", "") for ep in endpoints]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probs + [probs[0]],
        theta=short_names + [short_names[0]],
        fill="toself",
        fillcolor="rgba(224, 75, 74, 0.2)",
        line=dict(color="#E24B4A", width=2),
        name="Toxicity Risk",
    ))
    # Threshold rings
    for threshold, label, color in [(0.4, "Medium", "#EF9F27"), (0.7, "High", "#E24B4A")]:
        fig.add_trace(go.Scatterpolar(
            r=[threshold] * (len(endpoints) + 1),
            theta=short_names + [short_names[0]],
            mode="lines",
            line=dict(color=color, width=1, dash="dash"),
            name=f"{label} threshold",
            showlegend=True,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(color="#888", size=10),
                gridcolor="#2a2d3a",
            ),
            angularaxis=dict(
                tickfont=dict(color="#ccc", size=11),
                gridcolor="#2a2d3a",
            ),
            bgcolor="#1a1d27",
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        legend=dict(font=dict(color="#ccc"), bgcolor="#1a1d27"),
        margin=dict(l=60, r=60, t=40, b=40),
        height=420,
    )
    return fig


# ── SHAP bar chart ────────────────────────────────────────────────────────────

def make_shap_bar(shap_top, endpoint):
    names  = [f["name"].replace("morgan_", "MFP-")[:28] for f in shap_top]
    values = [f["shap"] for f in shap_top]
    colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=names[::-1],
        orientation="h",
        marker_color=colors[::-1],
    ))
    fig.update_layout(
        title=dict(text=f"SHAP contributions — {endpoint}", font=dict(color="#ccc", size=13)),
        xaxis=dict(title="SHAP value", color="#888", gridcolor="#2a2d3a", zeroline=True, zerolinecolor="#555"),
        yaxis=dict(color="#ccc", tickfont=dict(size=10)),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        height=320,
        margin=dict(l=10, r=10, t=40, b=30),
    )
    return fig


# ── Model performance chart ───────────────────────────────────────────────────

def make_auc_chart(payload):
    eps  = list(payload["aucs"].keys())
    aucs = [payload["aucs"][ep] for ep in eps]
    colors = ["#E24B4A" if a < 0.70 else "#EF9F27" if a < 0.80 else "#1D9E75" for a in aucs]

    fig = go.Figure(go.Bar(
        x=[ep.replace("NR-","").replace("SR-","") for ep in eps],
        y=aucs,
        marker_color=colors,
        text=[f"{a:.3f}" for a in aucs],
        textposition="outside",
        textfont=dict(color="#ccc", size=11),
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="#EF9F27",
                  annotation_text="0.70 baseline", annotation_font_color="#EF9F27")
    fig.update_layout(
        xaxis=dict(color="#888", tickangle=-30),
        yaxis=dict(range=[0.5, 1.0], color="#888", gridcolor="#2a2d3a"),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        height=300,
        margin=dict(l=10, r=10, t=10, b=60),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(payload):
    st.sidebar.markdown("## SafeDrug AI")
    st.sidebar.markdown("Multi-task toxicity prediction with SHAP-guided molecular insights.")
    st.sidebar.markdown("---")

    if payload:
        avg_auc = np.mean(list(payload["aucs"].values()))
        st.sidebar.metric("Models loaded", len(payload["models"]))
        st.sidebar.metric("Avg ROC-AUC", f"{avg_auc:.3f}")
        st.sidebar.markdown("---")

    st.sidebar.markdown("### Demo molecules")
    demo_choice = st.sidebar.selectbox("Load a demo", ["— custom input —"] + list(DEMO_MOLECULES.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About the 12 endpoints")
    if payload:
        for ep, (name, desc) in payload["endpoint_bio"].items():
            with st.sidebar.expander(ep):
                st.markdown(f"**{name}**")
                st.caption(desc)

    return demo_choice


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    payload = get_models()
    demo_choice = render_sidebar(payload)

    # Header
    st.markdown("""
    <h1 style='color:#ffffff; margin-bottom:4px;'>SafeDrug AI</h1>
    <p style='color:#888; margin-top:0;'>
    Multi-task toxicity prediction · SHAP explainability · Biological pathway analysis
    </p>
    """, unsafe_allow_html=True)

    if payload is None:
        st.error("Models not found. Please run `python train.py --data tox21.csv` first.")
        st.code("python train.py --data tox21.csv", language="bash")
        return

    # Input
    st.markdown('<div class="section-header">Molecule input</div>', unsafe_allow_html=True)

    default_smiles = ""
    if demo_choice != "— custom input —":
        default_smiles = DEMO_MOLECULES[demo_choice]

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        smiles = st.text_input(
            "SMILES string",
            value=default_smiles,
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
            label_visibility="collapsed",
        )
    with col_btn:
        analyse = st.button("Analyse", type="primary", use_container_width=True)

    if not smiles:
        st.info("Enter a SMILES string above or select a demo molecule from the sidebar.")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string. Please check your input.")
        return

    # Run prediction
    with st.spinner("Running multi-task prediction + SHAP analysis..."):
        output = predict_all_endpoints(smiles, payload)

    if output is None:
        st.error("Could not generate features for this molecule.")
        return

    results, rdkit_mol = output

    # ── Top summary row ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Toxicity summary</div>', unsafe_allow_html=True)

    high   = sum(1 for r in results.values() if r["risk_label"] == "High")
    medium = sum(1 for r in results.values() if r["risk_label"] == "Medium")
    low    = sum(1 for r in results.values() if r["risk_label"] == "Low")
    max_risk_ep = max(results, key=lambda e: results[e]["prob"])
    max_prob = results[max_risk_ep]["prob"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("High risk pathways",   high,   delta_color="inverse")
    col2.metric("Medium risk pathways", medium, delta_color="inverse")
    col3.metric("Low risk pathways",    low)
    col4.metric("Peak toxicity score",  f"{max_prob:.2f}", delta=max_risk_ep, delta_color="inverse")

    # ── Two-column layout: radar + molecule ──────────────────────────────────
    st.markdown('<div class="section-header">Toxicity fingerprint</div>', unsafe_allow_html=True)
    col_radar, col_mol = st.columns([3, 2])

    with col_radar:
        st.plotly_chart(make_radar_chart(results), use_container_width=True)
        st.caption("Each axis = one Tox21 biological assay. Dashed rings show medium/high thresholds.")

    with col_mol:
        st.markdown("**Molecule structure**")
        mol_img = Draw.MolToImage(rdkit_mol, size=(400, 300))
        st.image(mol_img, caption=f"SMILES: {smiles[:60]}{'...' if len(smiles)>60 else ''}")

        # Basic properties
        from rdkit.Chem import Descriptors as D
        props = {
            "Mol. weight": f"{D.MolWt(rdkit_mol):.1f} Da",
            "logP": f"{D.MolLogP(rdkit_mol):.2f}",
            "TPSA": f"{D.TPSA(rdkit_mol):.1f} Å²",
            "H-donors": str(D.NumHDonors(rdkit_mol)),
            "H-acceptors": str(D.NumHAcceptors(rdkit_mol)),
            "Rotatable bonds": str(D.NumRotatableBonds(rdkit_mol)),
        }
        prop_df = pd.DataFrame(props.items(), columns=["Property", "Value"])
        st.dataframe(prop_df, hide_index=True, use_container_width=True)

    # ── SHAP heatmap on molecule ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Atom-level toxicity heatmap</div>', unsafe_allow_html=True)

    ep_options = list(results.keys())
    ep_default = max(results, key=lambda e: results[e]["prob"])
    selected_ep = st.selectbox(
        "Select endpoint for heatmap",
        ep_options,
        index=ep_options.index(ep_default),
        format_func=lambda e: f"{e} — {results[e]['bio'][0]} ({results[e]['prob']:.0%} risk)"
    )

    heatmap = render_molecule_heatmap(smiles, selected_ep, payload)
    col_hm, col_bio = st.columns([2, 1])

    with col_hm:
        if isinstance(heatmap, str):  # SVG string
            st.components.v1.html(heatmap, height=420)
            st.caption("Red = toxic contribution, Green = protective")
        elif heatmap is not None:
            st.image(heatmap, caption="Red = toxic contribution, Green = protective")

    with col_bio:
        ep_data = results[selected_ep]
        pathway, mechanism = ep_data["bio"]
        rl, rc = ep_data["risk_label"], ep_data["risk_color"]
        st.markdown(f"### {selected_ep}")
        st.markdown(f"**Pathway:** {pathway}")
        st.markdown(f"**Mechanism:** {mechanism}")
        st.markdown(f"**Risk score:** <span style='color:{rc}; font-size:24px; font-weight:700;'>{ep_data['prob']:.1%}</span>", unsafe_allow_html=True)
        st.markdown(f"**Level:** <span style='color:{rc};'>{rl}</span>", unsafe_allow_html=True)
        st.markdown(f"**Model ROC-AUC:** {ep_data['auc']:.3f}")

    # ── Per-endpoint SHAP breakdown ──────────────────────────────────────────
    st.markdown('<div class="section-header">SHAP feature contributions — all endpoints</div>', unsafe_allow_html=True)

    tabs = st.tabs([ep.replace("NR-","NR·").replace("SR-","SR·") for ep in results.keys()])
    for tab, (ep, data) in zip(tabs, results.items()):
        with tab:
            col_shap, col_info = st.columns([3, 2])
            with col_shap:
                st.plotly_chart(make_shap_bar(data["shap_top"], ep), use_container_width=True)
            with col_info:
                st.markdown(f"**{ep}** — {data['bio'][0]}")
                st.caption(data["bio"][1])
                prob = data["prob"]
                label, color = data["risk_label"], data["risk_color"]
                st.markdown(
                    f"<div style='font-size:32px; font-weight:700; color:{color};'>{prob:.1%}</div>"
                    f"<div style='color:{color};'>{label} risk</div>",
                    unsafe_allow_html=True
                )
                st.markdown("**Top toxic drivers:**")
                for feat in data["shap_top"][:3]:
                    if feat["shap"] > 0:
                        fname = feat["name"].replace("morgan_", "Fingerprint bit ") 
                        st.markdown(f"- `{fname[:35]}` → +{feat['shap']:.4f}")

    # ── Modification suggestions ─────────────────────────────────────────────
    st.markdown('<div class="section-header">SHAP-guided modification suggestions</div>', unsafe_allow_html=True)

    suggestions = suggest_modifications(smiles, results, payload)
    if suggestions:
        for sug in suggestions:
            ep, pathway = sug["endpoint"], sug["pathway"]
            old_risk = sug["risk"]
            new_est  = sug["new_risk_est"]
            delta    = old_risk - new_est
            _, old_color = risk_level(old_risk)
            _, new_color = risk_level(new_est)

            st.markdown(f"""
            <div class="suggestion-card">
                <div style="font-weight:600; color:#ffffff; margin-bottom:4px;">
                    {ep} — {pathway}
                </div>
                <div style="font-size:13px; color:#aaa; margin-bottom:8px;">
                    Current risk: <span style="color:{old_color}; font-weight:600;">{old_risk:.1%}</span>
                    &nbsp;&nbsp;→&nbsp;&nbsp;
                    Est. after modification: <span style="color:{new_color}; font-weight:600;">{new_est:.1%}</span>
                    &nbsp;&nbsp;
                    <span style="color:#1D9E75;">(−{delta:.1%})</span>
                </div>
                <div style="font-size:14px; color:#e0e0e0;">
                    <strong>Suggestion:</strong> {sug['hint']}
                </div>
                <div style="font-size:12px; color:#666; margin-top:4px;">
                    Driven by: <code>{sug['feature'][:40]}</code> (SHAP = +{sug['shap']:.4f})
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No high-risk endpoints detected. Molecule appears safe across all tested pathways.")

    # ── Model performance ────────────────────────────────────────────────────
    with st.expander("Model performance (ROC-AUC per endpoint)"):
        st.plotly_chart(make_auc_chart(payload), use_container_width=True)
        auc_df = pd.DataFrame([
            {"Endpoint": ep, "Pathway": payload["endpoint_bio"][ep][0],
             "ROC-AUC": f"{auc:.4f}"}
            for ep, auc in payload["aucs"].items()
        ])
        st.dataframe(auc_df, hide_index=True, use_container_width=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "SafeDrug AI · Trained on Tox21 (7,831 compounds, 12 assay endpoints) · "
        "SHAP TreeExplainer for molecular interpretability · "
        "For research use only."
    )


if __name__ == "__main__":
    main()
