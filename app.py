"""
SafeDrug AI — Complete App
Streamlit dashboard. Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors as D
from PIL import Image
import io

from explainer import (
    load_models, predict_all_endpoints,
    render_molecule_heatmap, suggest_modifications,
    risk_level, lipinski_rules
)

st.set_page_config(
    page_title="SafeDrug AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .section-header {
        font-size: 18px; font-weight: 600; color: #ffffff;
        border-bottom: 1px solid #2a2d3a; padding-bottom: 8px; margin: 20px 0 12px;
    }
    .suggestion-card {
        background: #1e2433; border: 1px solid #2a3550;
        border-radius: 8px; padding: 14px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

DEMO_MOLECULES = {
    "Aspirin (safe — control)":         "CC(=O)Oc1ccccc1C(=O)O",
    "Tamoxifen (ER toxicity)":           "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Nitrobenzene (nitro group, toxic)": "O=[N+]([O-])c1ccccc1",
    "Caffeine (low toxicity)":           "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Bisphenol A (endocrine disruptor)": "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C",
}

@st.cache_resource
def get_models():
    try:
        return load_models("models/safedrug_models.pkl")
    except FileNotFoundError:
        return None

def make_radar_chart(results):
    endpoints   = list(results.keys())
    probs       = [results[ep]["prob"] for ep in endpoints]
    short_names = [ep.replace("NR-", "").replace("SR-", "") for ep in endpoints]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probs + [probs[0]], theta=short_names + [short_names[0]],
        fill="toself", fillcolor="rgba(224,75,74,0.2)",
        line=dict(color="#E24B4A", width=2), name="Toxicity Risk",
    ))
    for threshold, label, color in [(0.4, "Medium", "#EF9F27"), (0.7, "High", "#E24B4A")]:
        fig.add_trace(go.Scatterpolar(
            r=[threshold] * (len(endpoints) + 1), theta=short_names + [short_names[0]],
            mode="lines", line=dict(color=color, width=1, dash="dash"),
            name=f"{label} threshold",
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1],
                tickfont=dict(color="#888", size=10), gridcolor="#2a2d3a"),
            angularaxis=dict(tickfont=dict(color="#ccc", size=11), gridcolor="#2a2d3a"),
            bgcolor="#1a1d27",
        ),
        paper_bgcolor="#0f1117",
        legend=dict(font=dict(color="#ccc"), bgcolor="#1a1d27"),
        margin=dict(l=60, r=60, t=40, b=40), height=420,
    )
    return fig

def make_shap_bar(shap_top, endpoint):
    names  = [f["name"].replace("morgan_", "MFP-")[:28] for f in shap_top]
    values = [f["shap"] for f in shap_top]
    colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in values]
    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1],
        orientation="h", marker_color=colors[::-1],
    ))
    fig.update_layout(
        title=dict(text=f"SHAP contributions — {endpoint}", font=dict(color="#ccc", size=13)),
        xaxis=dict(title="SHAP value", color="#888", gridcolor="#2a2d3a",
                   zeroline=True, zerolinecolor="#555"),
        yaxis=dict(color="#ccc", tickfont=dict(size=10)),
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        height=320, margin=dict(l=10, r=10, t=40, b=30),
    )
    return fig

def make_auc_chart(payload):
    eps   = list(payload["aucs"].keys())
    aucs  = [payload["aucs"][ep] for ep in eps]
    colors = ["#E24B4A" if a < 0.70 else "#EF9F27" if a < 0.80 else "#1D9E75" for a in aucs]
    fig = go.Figure(go.Bar(
        x=[ep.replace("NR-","").replace("SR-","") for ep in eps], y=aucs,
        marker_color=colors, text=[f"{a:.3f}" for a in aucs],
        textposition="outside", textfont=dict(color="#ccc", size=11),
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color="#EF9F27",
                  annotation_text="0.70 baseline", annotation_font_color="#EF9F27")
    fig.update_layout(
        xaxis=dict(color="#888", tickangle=-30),
        yaxis=dict(range=[0.5, 1.0], color="#888", gridcolor="#2a2d3a"),
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        height=300, margin=dict(l=10, r=10, t=10, b=60),
    )
    return fig

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
    demo_choice = st.sidebar.selectbox(
        "Load a demo", ["— custom input —"] + list(DEMO_MOLECULES.keys())
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About the 12 endpoints")
    if payload:
        for ep, (name, desc) in payload["endpoint_bio"].items():
            with st.sidebar.expander(ep):
                st.markdown(f"**{name}**")
                st.caption(desc)
    return demo_choice

def render_lipinski(smiles):
    data = lipinski_rules(smiles)
    if data is None:
        return
    verdict = data["verdict"]
    core    = data["core_passes"]
    v_color = "#1D9E75" if verdict == "Excellent" else "#EF9F27" if verdict == "Acceptable" else "#E24B4A"
    st.markdown(
        f"""<div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;
                padding:16px 20px;margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        margin-bottom:14px;">
                <div>
                    <span style="font-size:15px;font-weight:600;color:#fff;">
                        Lipinski's Rule of Five</span>
                    <span style="font-size:12px;color:#666;margin-left:8px;">
                        Drug-likeness assessment</span>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:13px;font-weight:700;color:{v_color};
                        background:{v_color}22;padding:4px 12px;border-radius:20px;
                        border:1px solid {v_color}44;">{verdict}</span>
                    <div style="font-size:11px;color:#666;margin-top:3px;">
                        {core}/4 core rules passed</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">""",
        unsafe_allow_html=True
    )
    for r in data["rules"]:
        icon   = "✓" if r["pass"] else "✗"
        color  = "#1D9E75" if r["pass"] else "#E24B4A"
        bg     = "#0d2b1e" if r["pass"] else "#2b0d0d"
        border = "#1D9E7544" if r["pass"] else "#E24B4A44"
        st.markdown(
            f"""<div style="background:{bg};border:1px solid {border};
                    border-radius:8px;padding:10px 12px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:11px;color:#888;">{r['rule']}</span>
                    <span style="font-size:13px;font-weight:700;color:{color};">{icon}</span>
                </div>
                <div style="font-size:18px;font-weight:700;color:#fff;margin:4px 0;">
                    {r['value']}{r['unit']}</div>
                <div style="font-size:10px;color:#666;">Limit: {r['limit']}</div>
            </div>""",
            unsafe_allow_html=True
        )
    st.markdown("</div></div>", unsafe_allow_html=True)

def render_comparison_tab(payload):
    st.markdown(
        '<div class="section-header">Molecule comparison — original vs modified</div>',
        unsafe_allow_html=True
    )
    st.caption("Enter two molecules to compare their toxicity profiles side by side on the same radar.")

    # ── Initialise session state backing stores (never clash with widget keys) ──
    for key, default in [
        ("cmp_a_val", ""), ("cmp_b_val", ""),
        ("lbl_a_val", "Molecule A"), ("lbl_b_val", "Molecule B"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Quick demo buttons write to _val keys only ──
    st.markdown("**Quick demo pairs:**")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        if st.button("Nitrobenzene vs Phenol", use_container_width=True):
            st.session_state["cmp_a_val"] = "O=[N+]([O-])c1ccccc1"
            st.session_state["cmp_b_val"] = "Oc1ccccc1"
            st.session_state["lbl_a_val"] = "Nitrobenzene (toxic)"
            st.session_state["lbl_b_val"] = "Phenol (modified)"
            st.rerun()
    with qc2:
        if st.button("Tamoxifen vs Aspirin", use_container_width=True):
            st.session_state["cmp_a_val"] = "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"
            st.session_state["cmp_b_val"] = "CC(=O)Oc1ccccc1C(=O)O"
            st.session_state["lbl_a_val"] = "Tamoxifen"
            st.session_state["lbl_b_val"] = "Aspirin"
            st.rerun()
    with qc3:
        if st.button("BPA vs Bisphenol S", use_container_width=True):
            st.session_state["cmp_a_val"] = "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C"
            st.session_state["cmp_b_val"] = "Oc1ccc(S(=O)(=O)c2ccc(O)cc2)cc1"
            st.session_state["lbl_a_val"] = "Bisphenol A"
            st.session_state["lbl_b_val"] = "Bisphenol S (safer alt?)"
            st.rerun()

    # ── Inputs use widget keys (cmp_a / cmp_b) but seed value from _val store ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Molecule A** — original")
        smiles_a = st.text_input(
            "SMILES A",
            value=st.session_state["cmp_a_val"],
            placeholder="e.g. O=[N+]([O-])c1ccccc1",
            key="cmp_a",
        )
        label_a = st.text_input(
            "Label A",
            value=st.session_state["lbl_a_val"],
            key="lbl_a",
        )
    with col_b:
        st.markdown("**Molecule B** — modified / reference")
        smiles_b = st.text_input(
            "SMILES B",
            value=st.session_state["cmp_b_val"],
            placeholder="e.g. Oc1ccccc1",
            key="cmp_b",
        )
        label_b = st.text_input(
            "Label B",
            value=st.session_state["lbl_b_val"],
            key="lbl_b",
        )

    if not smiles_a or not smiles_b:
        st.info("Enter both SMILES strings above or click a quick demo pair.")
        return

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None:
        st.error("Molecule A: invalid SMILES")
        return
    if mol_b is None:
        st.error("Molecule B: invalid SMILES")
        return

    with st.spinner("Running comparison analysis..."):
        out_a = predict_all_endpoints(smiles_a, payload)
        out_b = predict_all_endpoints(smiles_b, payload)

    if out_a is None or out_b is None:
        st.error("Could not generate features for one or both molecules.")
        return

    results_a, _ = out_a
    results_b, _ = out_b

    # Molecule structures
    sc1, sc2 = st.columns(2)
    with sc1:
        st.image(Draw.MolToImage(mol_a, size=(400, 280)),
                 caption=label_a, use_container_width=True)
    with sc2:
        st.image(Draw.MolToImage(mol_b, size=(400, 280)),
                 caption=label_b, use_container_width=True)

    # Overlay radar
    endpoints   = list(results_a.keys())
    short_names = [ep.replace("NR-","").replace("SR-","") for ep in endpoints]
    probs_a = [results_a[ep]["prob"] for ep in endpoints]
    probs_b = [results_b[ep]["prob"] for ep in endpoints]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probs_a + [probs_a[0]], theta=short_names + [short_names[0]],
        fill="toself", fillcolor="rgba(224,75,74,0.15)",
        line=dict(color="#E24B4A", width=2.5), name=label_a,
    ))
    fig.add_trace(go.Scatterpolar(
        r=probs_b + [probs_b[0]], theta=short_names + [short_names[0]],
        fill="toself", fillcolor="rgba(55,138,221,0.15)",
        line=dict(color="#378ADD", width=2.5), name=label_b,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1],
                tickfont=dict(color="#888", size=10), gridcolor="#2a2d3a"),
            angularaxis=dict(tickfont=dict(color="#ccc", size=11), gridcolor="#2a2d3a"),
            bgcolor="#1a1d27",
        ),
        paper_bgcolor="#0f1117",
        legend=dict(font=dict(color="#ccc"), bgcolor="#1a1d27",
                    orientation="h", y=-0.15),
        margin=dict(l=60, r=60, t=20, b=60), height=460,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red = Molecule A · Blue = Molecule B · Larger area = more toxic")

    # Side-by-side endpoint summary
    st.markdown("**Endpoint-by-endpoint comparison**")
    col_l, col_r = st.columns(2)

    def _summary_col(col, results, label, color):
        with col:
            high   = sum(1 for r in results.values() if r["risk_label"] == "High")
            medium = sum(1 for r in results.values() if r["risk_label"] == "Medium")
            avg    = sum(r["prob"] for r in results.values()) / len(results)
            st.markdown(
                f"""<div style="background:#1a1d27;border:1px solid {color}44;
                    border-radius:10px;padding:14px;margin-bottom:10px;">
                    <div style="font-size:14px;font-weight:600;color:{color};
                        margin-bottom:8px;">{label}</div>
                    <div style="display:flex;gap:16px;font-size:13px;color:#ccc;">
                        <span>High: <strong style="color:#E24B4A;">{high}</strong></span>
                        <span>Medium: <strong style="color:#EF9F27;">{medium}</strong></span>
                        <span>Avg: <strong style="color:#fff;">{avg:.2f}</strong></span>
                    </div>
                </div>""",
                unsafe_allow_html=True
            )
            for ep in list(results.keys()):
                pa = results[ep]["prob"]
                _, rc = risk_level(pa)
                st.markdown(
                    f"""<div style="display:flex;justify-content:space-between;
                        align-items:center;padding:5px 0;
                        border-bottom:1px solid #1e2130;font-size:12px;">
                        <span style="color:#aaa;">{ep}</span>
                        <span style="color:{rc};font-weight:600;">{pa:.1%}</span>
                    </div>""",
                    unsafe_allow_html=True
                )

    _summary_col(col_l, results_a, label_a, "#E24B4A")
    _summary_col(col_r, results_b, label_b, "#378ADD")

    # Delta table
    deltas = []
    for ep in endpoints:
        delta = results_a[ep]["prob"] - results_b[ep]["prob"]
        deltas.append({
            "Endpoint": ep,
            "Pathway":  results_a[ep]["bio"][0],
            label_a:    f"{results_a[ep]['prob']:.1%}",
            label_b:    f"{results_b[ep]['prob']:.1%}",
            "Risk reduction": f"{delta:+.1%}",
        })
    df_delta = pd.DataFrame(deltas).sort_values("Risk reduction", ascending=False)
    st.markdown("**Risk delta — sorted by biggest reduction**")
    st.dataframe(df_delta, hide_index=True, use_container_width=True)
    st.caption(f"Positive = {label_a} is more toxic · Negative = {label_b} is more toxic")

    # Lipinski comparison
    st.markdown("**Drug-likeness comparison (Lipinski)**")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.markdown(f"*{label_a}*")
        render_lipinski(smiles_a)
    with lc2:
        st.markdown(f"*{label_b}*")
        render_lipinski(smiles_b)


def render_batch_tab(payload):
    st.markdown(
        '<div class="section-header">Batch screening — screen multiple molecules at once</div>',
        unsafe_allow_html=True
    )
    st.caption("Upload a CSV with a 'smiles' column (and optional 'name' column). "
               "Get back a ranked toxicity table.")

    sample_csv = (
        "name,smiles\n"
        "Aspirin,CC(=O)Oc1ccccc1C(=O)O\n"
        "Nitrobenzene,O=[N+]([O-])c1ccccc1\n"
        "Tamoxifen,CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1\n"
        "Caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C\n"
        "Bisphenol A,CC(c1ccc(O)cc1)(c1ccc(O)cc1)C\n"
        "Phenol,Oc1ccccc1\n"
        "Benzene,c1ccccc1\n"
        "Aniline,Nc1ccccc1\n"
    )
    st.download_button(
        "Download sample CSV", data=sample_csv,
        file_name="sample_molecules.csv", mime="text/csv"
    )

    uploaded   = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use sample molecules instead of uploading")

    if not uploaded and not use_sample:
        st.info("Upload a CSV or check 'Use sample molecules' to try it out.")
        return

    df_in = pd.read_csv(io.StringIO(sample_csv)) if use_sample else pd.read_csv(uploaded)

    if "smiles" not in df_in.columns:
        st.error("CSV must have a 'smiles' column.")
        return

    has_name = "name" in df_in.columns
    total    = len(df_in)
    st.info(f"Found {total} molecules. Running predictions...")

    progress = st.progress(0)
    status   = st.empty()
    rows     = []

    for i, row in df_in.iterrows():
        smi  = str(row["smiles"]).strip()
        name = str(row["name"]) if has_name else smi[:20]
        status.text(f"Analysing {i+1}/{total}: {name}")
        progress.progress((i + 1) / total)

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({
                "Name": name, "SMILES": smi, "Status": "Invalid SMILES",
                "Overall Risk": "—", "High Endpoints": 0, "Medium Endpoints": 0,
                "Avg Toxicity Score": 0.0, "Peak Endpoint": "—",
                "Peak Score": 0.0, "Lipinski": "—",
            })
            continue

        out = predict_all_endpoints(smi, payload)
        if out is None:
            rows.append({
                "Name": name, "SMILES": smi, "Status": "Feature error",
                "Overall Risk": "—", "High Endpoints": 0, "Medium Endpoints": 0,
                "Avg Toxicity Score": 0.0, "Peak Endpoint": "—",
                "Peak Score": 0.0, "Lipinski": "—",
            })
            continue

        results, _ = out
        high    = sum(1 for r in results.values() if r["risk_label"] == "High")
        medium  = sum(1 for r in results.values() if r["risk_label"] == "Medium")
        avg_tox = sum(r["prob"] for r in results.values()) / len(results)
        peak_ep = max(results, key=lambda e: results[e]["prob"])
        peak_sc = results[peak_ep]["prob"]
        overall = "High" if high >= 3 else "Medium" if high >= 1 or medium >= 3 else "Low"
        lip     = lipinski_rules(smi)

        rows.append({
            "Name": name, "SMILES": smi, "Status": "OK",
            "Overall Risk": overall,
            "High Endpoints": high, "Medium Endpoints": medium,
            "Avg Toxicity Score": round(avg_tox, 3),
            "Peak Endpoint": peak_ep, "Peak Score": round(peak_sc, 3),
            "Lipinski": lip["verdict"] if lip else "—",
        })

    progress.empty()
    status.empty()

    df_out = pd.DataFrame(rows).sort_values("Avg Toxicity Score", ascending=False)
    valid  = df_out[df_out["Status"] == "OK"]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total screened", len(df_out))
    s2.metric("High risk",      len(valid[valid["Overall Risk"] == "High"]))
    s3.metric("Medium risk",    len(valid[valid["Overall Risk"] == "Medium"]))
    s4.metric("Low risk",       len(valid[valid["Overall Risk"] == "Low"]))

    risk_counts = valid["Overall Risk"].value_counts()
    fig_risk = go.Figure(go.Bar(
        x=list(risk_counts.index), y=list(risk_counts.values),
        marker_color=[
            "#E24B4A" if r == "High" else "#EF9F27" if r == "Medium" else "#1D9E75"
            for r in risk_counts.index
        ],
        text=list(risk_counts.values),
        textposition="outside", textfont=dict(color="#ccc"),
    ))
    fig_risk.update_layout(
        xaxis=dict(color="#888"), yaxis=dict(color="#888", gridcolor="#2a2d3a"),
        paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        height=220, margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("**Ranked results — sorted by average toxicity score**")
    display_cols = [
        "Name", "Overall Risk", "High Endpoints", "Medium Endpoints",
        "Avg Toxicity Score", "Peak Endpoint", "Peak Score", "Lipinski",
    ]

    def color_risk(val):
        if val == "High":       return "color: #E24B4A; font-weight: bold"
        if val == "Medium":     return "color: #EF9F27; font-weight: bold"
        if val == "Low":        return "color: #1D9E75; font-weight: bold"
        if val == "Excellent":  return "color: #1D9E75"
        if val == "Acceptable": return "color: #EF9F27"
        if val == "Poor":       return "color: #E24B4A"
        return ""

    st.dataframe(
        df_out[display_cols].style.applymap(
            color_risk, subset=["Overall Risk", "Lipinski"]
        ),
        hide_index=True, use_container_width=True, height=400,
    )
    st.download_button(
        "Download full results CSV",
        data=df_out.to_csv(index=False),
        file_name="safedrug_batch_results.csv",
        mime="text/csv",
    )


def render_single_tab(payload, demo_choice):
    default_smiles = (
        DEMO_MOLECULES.get(demo_choice, "")
        if demo_choice != "— custom input —" else ""
    )
    st.markdown('<div class="section-header">Molecule input</div>', unsafe_allow_html=True)
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        smiles = st.text_input(
            "SMILES string", value=default_smiles,
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
            label_visibility="collapsed",
        )
    with col_btn:
        st.button("Analyse", type="primary", use_container_width=True)

    if not smiles:
        st.info("Enter a SMILES string above or select a demo molecule from the sidebar.")
        return

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string. Please check your input.")
        return

    with st.spinner("Running multi-task prediction + SHAP analysis..."):
        output = predict_all_endpoints(smiles, payload)

    if output is None:
        st.error("Could not generate features for this molecule.")
        return

    results, rdkit_mol = output

    # Summary metrics
    st.markdown('<div class="section-header">Toxicity summary</div>', unsafe_allow_html=True)
    high   = sum(1 for r in results.values() if r["risk_label"] == "High")
    medium = sum(1 for r in results.values() if r["risk_label"] == "Medium")
    low    = sum(1 for r in results.values() if r["risk_label"] == "Low")
    max_ep = max(results, key=lambda e: results[e]["prob"])
    max_pb = results[max_ep]["prob"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High risk pathways",   high,   delta_color="inverse")
    c2.metric("Medium risk pathways", medium, delta_color="inverse")
    c3.metric("Low risk pathways",    low)
    c4.metric("Peak toxicity score",  f"{max_pb:.2f}", delta=max_ep, delta_color="inverse")

    # Lipinski
    st.markdown(
        '<div class="section-header">Drug-likeness (Lipinski\'s Rule of Five)</div>',
        unsafe_allow_html=True
    )
    render_lipinski(smiles)

    # Radar + molecule
    st.markdown('<div class="section-header">Toxicity fingerprint</div>', unsafe_allow_html=True)
    col_radar, col_mol = st.columns([3, 2])
    with col_radar:
        st.plotly_chart(make_radar_chart(results), use_container_width=True)
        st.caption("Each axis = one Tox21 biological assay. Dashed rings show medium/high thresholds.")
    with col_mol:
        st.markdown("**Molecule structure**")
        st.image(
            Draw.MolToImage(rdkit_mol, size=(400, 300)),
            caption=f"SMILES: {smiles[:60]}{'...' if len(smiles) > 60 else ''}",
        )
        props = {
            "Mol. weight":     f"{D.MolWt(rdkit_mol):.1f} Da",
            "logP":            f"{D.MolLogP(rdkit_mol):.2f}",
            "TPSA":            f"{D.TPSA(rdkit_mol):.1f} Å²",
            "H-donors":        str(D.NumHDonors(rdkit_mol)),
            "H-acceptors":     str(D.NumHAcceptors(rdkit_mol)),
            "Rotatable bonds": str(D.NumRotatableBonds(rdkit_mol)),
        }
        st.dataframe(
            pd.DataFrame(props.items(), columns=["Property", "Value"]),
            hide_index=True, use_container_width=True,
        )

    # Atom heatmap
    st.markdown(
        '<div class="section-header">Atom-level toxicity heatmap</div>',
        unsafe_allow_html=True
    )
    ep_options = list(results.keys())
    ep_default = max(results, key=lambda e: results[e]["prob"])
    selected_ep = st.selectbox(
        "Select endpoint for heatmap", ep_options,
        index=ep_options.index(ep_default),
        format_func=lambda e: f"{e} — {results[e]['bio'][0]} ({results[e]['prob']:.0%} risk)",
    )
    heatmap = render_molecule_heatmap(smiles, selected_ep, payload)
    col_hm, col_bio = st.columns([2, 1])
    with col_hm:
        if isinstance(heatmap, tuple) and heatmap[0] == "html":
            st.components.v1.html(heatmap[1], height=450)
        elif isinstance(heatmap, str):
            st.components.v1.html(
                f'<div style="background:#fff;border-radius:8px;padding:8px;">{heatmap}</div>',
                height=450,
            )
        elif heatmap is not None:
            st.image(heatmap, caption="Red = toxic contribution, Green = protective")
    with col_bio:
        ep_data = results[selected_ep]
        pathway, mechanism = ep_data["bio"]
        rl, rc = ep_data["risk_label"], ep_data["risk_color"]
        st.markdown(f"### {selected_ep}")
        st.markdown(f"**Pathway:** {pathway}")
        st.markdown(f"**Mechanism:** {mechanism}")
        st.markdown(
            f"**Risk score:** <span style='color:{rc};font-size:24px;font-weight:700;'>"
            f"{ep_data['prob']:.1%}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Level:** <span style='color:{rc};'>{rl}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Model ROC-AUC:** {ep_data['auc']:.3f}")

    # SHAP tabs
    st.markdown(
        '<div class="section-header">SHAP feature contributions — all endpoints</div>',
        unsafe_allow_html=True
    )
    tabs = st.tabs([ep.replace("NR-","NR·").replace("SR-","SR·") for ep in results.keys()])
    for tab, (ep, data) in zip(tabs, results.items()):
        with tab:
            col_shap, col_info = st.columns([3, 2])
            with col_shap:
                st.plotly_chart(make_shap_bar(data["shap_top"], ep), use_container_width=True)
            with col_info:
                st.markdown(f"**{ep}** — {data['bio'][0]}")
                st.caption(data["bio"][1])
                prob  = data["prob"]
                label, color = data["risk_label"], data["risk_color"]
                st.markdown(
                    f"<div style='font-size:32px;font-weight:700;color:{color};'>{prob:.1%}</div>"
                    f"<div style='color:{color};'>{label} risk</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Top toxic drivers:**")
                for feat in data["shap_top"][:3]:
                    if feat["shap"] > 0:
                        fname = feat["name"].replace("morgan_", "Fingerprint bit ")
                        st.markdown(f"- `{fname[:35]}` → +{feat['shap']:.4f}")

    # Modification suggestions
    st.markdown(
        '<div class="section-header">SHAP-guided modification suggestions</div>',
        unsafe_allow_html=True
    )
    suggestions = suggest_modifications(smiles, results, payload)
    if suggestions:
        for sug in suggestions:
            old_risk = sug["risk"]
            new_est  = sug["new_risk_est"]
            delta    = old_risk - new_est
            _, oc = risk_level(old_risk)
            _, nc = risk_level(new_est)
            st.markdown(
                f"""<div class="suggestion-card">
                    <div style="font-weight:600;color:#fff;margin-bottom:4px;">
                        {sug['endpoint']} — {sug['pathway']}</div>
                    <div style="font-size:13px;color:#aaa;margin-bottom:8px;">
                        Current: <span style="color:{oc};font-weight:600;">{old_risk:.1%}</span>
                        &nbsp;→&nbsp;
                        After modification:
                        <span style="color:{nc};font-weight:600;">{new_est:.1%}</span>
                        &nbsp;<span style="color:#1D9E75;">(−{delta:.1%})</span>
                    </div>
                    <div style="font-size:14px;color:#e0e0e0;">
                        <strong>Suggestion:</strong> {sug['hint']}</div>
                    <div style="font-size:12px;color:#666;margin-top:4px;">
                        Driven by: <code>{sug['feature'][:40]}</code>
                        (SHAP = +{sug['shap']:.4f})
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.success("No high-risk endpoints detected. Molecule appears safe across all tested pathways.")

    # Model performance
    with st.expander("Model performance (ROC-AUC per endpoint)"):
        st.plotly_chart(make_auc_chart(payload), use_container_width=True)
        st.dataframe(
            pd.DataFrame([
                {"Endpoint": ep,
                 "Pathway":  payload["endpoint_bio"][ep][0],
                 "ROC-AUC":  f"{auc:.4f}"}
                for ep, auc in payload["aucs"].items()
            ]),
            hide_index=True, use_container_width=True,
        )

    st.markdown("---")
    st.caption(
        "SafeDrug AI · Trained on Tox21 (7,831 compounds, 12 assay endpoints) · "
        "SHAP TreeExplainer for molecular interpretability · For research use only."
    )


def main():
    payload     = get_models()
    demo_choice = render_sidebar(payload)

    st.markdown("""
    <h1 style='color:#ffffff;margin-bottom:4px;'>SafeDrug AI</h1>
    <p style='color:#888;margin-top:0;'>
    Multi-task toxicity prediction · SHAP explainability · Biological pathway analysis
    </p>""", unsafe_allow_html=True)

    if payload is None:
        st.error("Models not found. Please run `python train.py --data tox21.csv` first.")
        st.code("python train.py --data tox21.csv", language="bash")
        return

    tab_single, tab_compare, tab_batch = st.tabs([
        "Single molecule", "Compare molecules", "Batch screening"
    ])
    with tab_single:
        render_single_tab(payload, demo_choice)
    with tab_compare:
        render_comparison_tab(payload)
    with tab_batch:
        render_batch_tab(payload)


if __name__ == "__main__":
    main()