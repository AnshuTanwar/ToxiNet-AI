

import io
import math
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import (
    HexColor, white, black, Color
)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Wedge, Circle
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.piecharts import Pie
from reportlab.platypus import KeepTogether

import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image as PILImage
from rdkit import Chem
from rdkit.Chem import Draw

# ── Colour palette ────────────────────────────────────────────────────────────

C_BG        = HexColor("#0f1117")
C_CARD      = HexColor("#1a1d27")
C_BORDER    = HexColor("#2a2d3a")
C_TEXT      = HexColor("#e0e0e0")
C_MUTED     = HexColor("#888888")
C_RED       = HexColor("#E24B4A")
C_AMBER     = HexColor("#EF9F27")
C_GREEN     = HexColor("#1D9E75")
C_BLUE      = HexColor("#378ADD")
C_WHITE     = white
C_DARK      = HexColor("#13151f")

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

# ── Styles ────────────────────────────────────────────────────────────────────

def _styles():
    return {
        "title": ParagraphStyle(
            "title", fontName="Helvetica-Bold", fontSize=22,
            textColor=C_WHITE, spaceAfter=4, leading=26,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontName="Helvetica", fontSize=10,
            textColor=C_MUTED, spaceAfter=12, leading=14,
        ),
        "section": ParagraphStyle(
            "section", fontName="Helvetica-Bold", fontSize=12,
            textColor=C_WHITE, spaceBefore=14, spaceAfter=6,
            leading=16, borderPad=4,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=9,
            textColor=C_TEXT, leading=13, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small", fontName="Helvetica", fontSize=8,
            textColor=C_MUTED, leading=11,
        ),
        "mono": ParagraphStyle(
            "mono", fontName="Courier", fontSize=8,
            textColor=HexColor("#7eb8f7"), leading=11,
        ),
        "risk_high": ParagraphStyle(
            "risk_high", fontName="Helvetica-Bold", fontSize=10,
            textColor=C_RED, leading=13,
        ),
        "risk_medium": ParagraphStyle(
            "risk_medium", fontName="Helvetica-Bold", fontSize=10,
            textColor=C_AMBER, leading=13,
        ),
        "risk_low": ParagraphStyle(
            "risk_low", fontName="Helvetica-Bold", fontSize=10,
            textColor=C_GREEN, leading=13,
        ),
    }

# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_color(label):
    return {"High": C_RED, "Medium": C_AMBER, "Low": C_GREEN}.get(label, C_MUTED)

def _risk_style(label, styles):
    return {"High": styles["risk_high"],
            "Medium": styles["risk_medium"],
            "Low": styles["risk_low"]}.get(label, styles["body"])

def _mol_image_bytes(smiles, size=(340, 260)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def _radar_image_bytes(results, width=480, height=380):
    """Render the toxicity radar chart as PNG bytes via Plotly."""
    endpoints   = list(results.keys())
    probs       = [results[ep]["prob"] for ep in endpoints]
    short_names = [ep.replace("NR-","").replace("SR-","") for ep in endpoints]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=probs + [probs[0]],
        theta=short_names + [short_names[0]],
        fill="toself",
        fillcolor="rgba(224,75,74,0.25)",
        line=dict(color="#E24B4A", width=2.5),
        name="Toxicity Risk",
    ))
    for threshold, color in [(0.4, "#EF9F27"), (0.7, "#E24B4A")]:
        fig.add_trace(go.Scatterpolar(
            r=[threshold] * (len(endpoints) + 1),
            theta=short_names + [short_names[0]],
            mode="lines",
            line=dict(color=color, width=1, dash="dash"),
            showlegend=False,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                tickfont=dict(color="#aaa", size=9), gridcolor="#333"),
            angularaxis=dict(tickfont=dict(color="#ccc", size=10), gridcolor="#333"),
            bgcolor="#1a1d27",
        ),
        paper_bgcolor="#13151f",
        margin=dict(l=50, r=50, t=30, b=30),
        width=width, height=height,
        showlegend=False,
    )
    png = pio.to_image(fig, format="png", width=width, height=height, scale=2)
    return io.BytesIO(png)

def _shap_bar_bytes(shap_top, endpoint, width=480, height=280):
    """Render a SHAP bar chart as PNG bytes."""
    names  = [f["name"].replace("morgan_", "MFP-")[:30] for f in shap_top[:8]]
    values = [f["shap"] for f in shap_top[:8]]
    colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1],
        orientation="h",
        marker_color=colors[::-1],
        marker_line_width=0,
    ))
    fig.update_layout(
        xaxis=dict(color="#888", gridcolor="#2a2d3a", zeroline=True, zerolinecolor="#555",
                title=dict(text="SHAP value", font=dict(color="#888", size=10))),
        yaxis=dict(color="#ccc", tickfont=dict(size=9)),
        paper_bgcolor="#13151f",
        plot_bgcolor="#1a1d27",
        width=width, height=height,
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
    )
    png = pio.to_image(fig, format="png", width=width, height=height, scale=2)
    return io.BytesIO(png)

# ── Section builders ──────────────────────────────────────────────────────────

def _build_header(smiles, results, lip_data, styles):
    """Top header block: title, SMILES, summary badges."""
    elements = []

    # Title row
    elements.append(Paragraph("SafeDrug AI", styles["title"]))
    elements.append(Paragraph(
        f"Toxicity Analysis Report  ·  Generated {datetime.now().strftime('%d %b %Y, %H:%M')}",
        styles["subtitle"]
    ))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=C_BORDER, spaceAfter=10
    ))

    # SMILES
    elements.append(Paragraph("Input molecule", styles["section"]))
    elements.append(Paragraph(smiles, styles["mono"]))
    elements.append(Spacer(1, 6))

    # Summary metrics row
    high   = sum(1 for r in results.values() if r["risk_label"] == "High")
    medium = sum(1 for r in results.values() if r["risk_label"] == "Medium")
    low    = sum(1 for r in results.values() if r["risk_label"] == "Low")
    max_ep = max(results, key=lambda e: results[e]["prob"])
    max_pb = results[max_ep]["prob"]
    _, max_color = max_ep, _risk_color(results[max_ep]["risk_label"])

    def _badge(label, value, color):
        return [
            Paragraph(str(value), ParagraphStyle(
                "bv", fontName="Helvetica-Bold", fontSize=20,
                textColor=color, leading=24, alignment=TA_CENTER
            )),
            Paragraph(label, ParagraphStyle(
                "bl", fontName="Helvetica", fontSize=8,
                textColor=C_MUTED, leading=11, alignment=TA_CENTER
            )),
        ]

    badge_data = [[
        _badge("High risk", high, C_RED),
        _badge("Medium risk", medium, C_AMBER),
        _badge("Low risk", low, C_GREEN),
        _badge(f"Peak ({max_ep})", f"{max_pb:.0%}", max_color),
    ]]

    col_w = CONTENT_W / 4
    badge_table = Table(badge_data, colWidths=[col_w]*4)
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_CARD),
        ("BOX",        (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    elements.append(badge_table)
    elements.append(Spacer(1, 12))
    return elements


def _build_molecule_and_radar(smiles, results, styles):
    """Two-column: molecule image left, radar chart right."""
    elements = []
    elements.append(Paragraph("Molecule structure + toxicity fingerprint", styles["section"]))

    mol_buf   = _mol_image_bytes(smiles, size=(320, 240))
    radar_buf = _radar_image_bytes(results, width=400, height=320)

    left_cell  = RLImage(mol_buf,   width=7.5*cm, height=5.6*cm) if mol_buf else Paragraph("Invalid SMILES", styles["body"])
    right_cell = RLImage(radar_buf, width=9.0*cm, height=7.2*cm)

    table = Table([[left_cell, right_cell]],
                  colWidths=[CONTENT_W * 0.45, CONTENT_W * 0.55])
    table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
        ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0), (-1,-1), 0.5, C_BORDER),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))
    return elements


def _build_lipinski(lip_data, styles):
    """Lipinski Rule of Five table."""
    if lip_data is None:
        return []
    elements = []
    elements.append(Paragraph("Drug-likeness — Lipinski's Rule of Five", styles["section"]))

    verdict  = lip_data["verdict"]
    v_color  = {"Excellent": C_GREEN, "Acceptable": C_AMBER}.get(verdict, C_RED)
    core     = lip_data["core_passes"]

    header = [[
        Paragraph("Rule", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12)),
        Paragraph("Value", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12, alignment=TA_CENTER)),
        Paragraph("Limit", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12, alignment=TA_CENTER)),
        Paragraph("Pass", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12, alignment=TA_CENTER)),
    ]]
    rows = []
    for r in lip_data["rules"]:
        pass_color = C_GREEN if r["pass"] else C_RED
        icon       = "✓" if r["pass"] else "✗"
        rows.append([
            Paragraph(r["rule"], styles["body"]),
            Paragraph(f"{r['value']}{r['unit']}",
                ParagraphStyle("cv", fontName="Helvetica-Bold", fontSize=9,
                    textColor=C_WHITE, leading=12, alignment=TA_CENTER)),
            Paragraph(r["limit"], styles["small"]),
            Paragraph(icon, ParagraphStyle("ic", fontName="Helvetica-Bold",
                fontSize=11, textColor=pass_color, leading=13, alignment=TA_CENTER)),
        ])

    verdict_row = [[
        Paragraph(f"Overall verdict: {verdict}  ({core}/4 core rules passed)",
            ParagraphStyle("vr", fontName="Helvetica-Bold", fontSize=10,
                textColor=v_color, leading=13)),
        "", "", "",
    ]]

    col_w = [CONTENT_W * 0.40, CONTENT_W * 0.20,
             CONTENT_W * 0.25, CONTENT_W * 0.15]
    t = Table(header + rows + verdict_row, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  HexColor("#1e2433")),
        ("BACKGROUND",    (0,1), (-1,-2), C_CARD),
        ("BACKGROUND",    (0,-1),(-1,-1), HexColor("#1e2433")),
        ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0), (-1,-2), 0.5, C_BORDER),
        ("SPAN",          (0,-1),(-1,-1)),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    return elements


def _build_endpoint_table(results, styles):
    """Full 12-endpoint risk table."""
    elements = []
    elements.append(Paragraph("Multi-task toxicity predictions — all 12 endpoints", styles["section"]))

    header = [[
        Paragraph("Endpoint", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12)),
        Paragraph("Biological pathway", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12)),
        Paragraph("Risk score", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12, alignment=TA_CENTER)),
        Paragraph("Level", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12, alignment=TA_CENTER)),
        Paragraph("Mechanism", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12)),
    ]]

    rows = []
    for ep, data in sorted(results.items(), key=lambda x: x[1]["prob"], reverse=True):
        pathway, mechanism = data["bio"]
        rc = _risk_color(data["risk_label"])
        rows.append([
            Paragraph(ep, ParagraphStyle("ep", fontName="Courier-Bold",
                fontSize=8, textColor=HexColor("#7eb8f7"), leading=11)),
            Paragraph(pathway, styles["body"]),
            Paragraph(f"{data['prob']:.1%}",
                ParagraphStyle("sc", fontName="Helvetica-Bold", fontSize=10,
                    textColor=rc, leading=13, alignment=TA_CENTER)),
            Paragraph(data["risk_label"],
                ParagraphStyle("rl", fontName="Helvetica-Bold", fontSize=9,
                    textColor=rc, leading=12, alignment=TA_CENTER)),
            Paragraph(mechanism[:55] + ("…" if len(mechanism) > 55 else ""),
                styles["small"]),
        ])

    col_w = [CONTENT_W*0.13, CONTENT_W*0.22, CONTENT_W*0.12,
             CONTENT_W*0.12, CONTENT_W*0.41]
    t = Table(header + rows, colWidths=col_w)
    row_styles = [
        ("BACKGROUND",    (0,0),  (-1,0),  HexColor("#1e2433")),
        ("BOX",           (0,0),  (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0),  (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0),  (-1,-1), 6),
        ("BOTTOMPADDING", (0,0),  (-1,-1), 6),
        ("LEFTPADDING",   (0,0),  (-1,-1), 8),
        ("RIGHTPADDING",  (0,0),  (-1,-1), 8),
        ("VALIGN",        (0,0),  (-1,-1), "MIDDLE"),
    ]
    for i, (ep, data) in enumerate(
        sorted(results.items(), key=lambda x: x[1]["prob"], reverse=True), start=1
    ):
        bg = HexColor("#1e0d0d") if data["risk_label"] == "High" else \
             HexColor("#1e1a0d") if data["risk_label"] == "Medium" else C_CARD
        row_styles.append(("BACKGROUND", (0,i), (-1,i), bg))
    t.setStyle(TableStyle(row_styles))
    elements.append(t)
    elements.append(Spacer(1, 12))
    return elements


def _build_shap_section(results, styles):
    """SHAP chart for the highest-risk endpoint."""
    elements = []
    top_ep   = max(results, key=lambda e: results[e]["prob"])
    data     = results[top_ep]
    elements.append(Paragraph(
        f"SHAP feature contributions — {top_ep} (highest risk endpoint)",
        styles["section"]
    ))

    shap_buf = _shap_bar_bytes(data["shap_top"], top_ep, width=560, height=260)
    img      = RLImage(shap_buf, width=CONTENT_W, height=CONTENT_W * 0.42)

    caption = Paragraph(
        "Red bars = features that increase toxicity risk · "
        "Green bars = features that reduce toxicity risk · "
        "MFP = Morgan Fingerprint bit",
        styles["small"]
    )
    t = Table([[img]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
        ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 4))
    elements.append(caption)
    elements.append(Spacer(1, 12))
    return elements


def _build_modifications(suggestions, styles):
    """Modification suggestions table."""
    if not suggestions:
        return []
    elements = []
    elements.append(Paragraph("SHAP-guided structural modification suggestions", styles["section"]))

    header = [[
        Paragraph("Endpoint", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12)),
        Paragraph("Current risk", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12, alignment=TA_CENTER)),
        Paragraph("Est. after", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12, alignment=TA_CENTER)),
        Paragraph("Reduction", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12, alignment=TA_CENTER)),
        Paragraph("Suggestion", ParagraphStyle("th", fontName="Helvetica-Bold",
            fontSize=9, textColor=C_MUTED, leading=12)),
    ]]
    rows = []
    for sug in suggestions:
        delta = sug["risk"] - sug["new_risk_est"]
        rows.append([
            Paragraph(sug["endpoint"], ParagraphStyle("ep2", fontName="Courier-Bold",
                fontSize=8, textColor=HexColor("#7eb8f7"), leading=11)),
            Paragraph(f"{sug['risk']:.1%}",
                ParagraphStyle("cr", fontName="Helvetica-Bold", fontSize=9,
                    textColor=C_RED, leading=12, alignment=TA_CENTER)),
            Paragraph(f"{sug['new_risk_est']:.1%}",
                ParagraphStyle("nr", fontName="Helvetica-Bold", fontSize=9,
                    textColor=C_GREEN, leading=12, alignment=TA_CENTER)),
            Paragraph(f"−{delta:.1%}",
                ParagraphStyle("dr", fontName="Helvetica-Bold", fontSize=9,
                    textColor=C_GREEN, leading=12, alignment=TA_CENTER)),
            Paragraph(sug["hint"], styles["small"]),
        ])

    col_w = [CONTENT_W*0.12, CONTENT_W*0.12, CONTENT_W*0.12,
             CONTENT_W*0.12, CONTENT_W*0.52]
    t = Table(header + rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  HexColor("#1e2433")),
        ("BACKGROUND",    (0,1), (-1,-1), C_CARD),
        ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))
    return elements


def _build_footer(styles):
    elements = []
    elements.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceBefore=6))
    elements.append(Paragraph(
        "ToxiNet AI  ·  Trained on Tox21 (7,831 compounds, 12 assay endpoints)  ·  "
        "SHAP TreeExplainer for molecular interpretability  ·  For research use only.",
        styles["small"]
    ))
    return elements


# ── Main export function ──────────────────────────────────────────────────────

def generate_pdf_report(smiles, results, lip_data, suggestions):
    """
    Generate a complete PDF report.
    Returns bytes object ready for st.download_button.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title="ToxiNet AI Report",
        author="ToxiNet AI",
    )

    styles   = _styles()
    story    = []

    # Dark background canvas via onPage callback
    def dark_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_BG)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.restoreState()

    story += _build_header(smiles, results, lip_data, styles)
    story += _build_molecule_and_radar(smiles, results, styles)
    story += _build_lipinski(lip_data, styles)
    story += _build_endpoint_table(results, styles)
    story += _build_shap_section(results, styles)
    story += _build_modifications(suggestions, styles)
    story += _build_footer(styles)

    doc.build(story, onFirstPage=dark_bg, onLaterPages=dark_bg)
    buf.seek(0)
    return buf.read()