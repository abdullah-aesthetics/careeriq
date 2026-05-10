"""
CareerIQ — PDF Report Generator (FIXED VERSION)
===============================================
Safe ReportLab implementation (no Flowable overflow crash)
"""

import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.platypus.flowables import Flowable

PAGE_W, PAGE_H = A4

# ─── COLORS ─────────────────────────────────────────────
C_DARK   = colors.HexColor('#111118')
C_CARD   = colors.HexColor('#16161f')
C_CARD2  = colors.HexColor('#1e1e2e')
C_BORDER = colors.HexColor('#2a2a3f')
C_PURPLE = colors.HexColor('#7c6aff')
C_PURP2  = colors.HexColor('#a78bfa')
C_TEAL   = colors.HexColor('#14b8a6')
C_GOLD   = colors.HexColor('#f59e0b')
C_SKY    = colors.HexColor('#38bdf8')
C_ROSE   = colors.HexColor('#f43f5e')
C_GREEN  = colors.HexColor('#22c55e')
C_WHITE  = colors.white
C_TEXT2  = colors.HexColor('#a9a9c8')
C_TEXT3  = colors.HexColor('#6b6b88')


# ─── SAFE FLOWABLE BASE ─────────────────────────────────
class SafeFlowable(Flowable):
    def wrap(self, *args):
        return getattr(self, "w", 0), getattr(self, "h", 20)


# ─── HEADER ─────────────────────────────────────────────
class HeaderBanner(SafeFlowable):
    def __init__(self, w, pred_id, ts):
        self.w = w
        self.h = 106
        self.pred_id = pred_id
        self.ts = ts

    def draw(self):
        c = self.canv
        c.setFillColor(C_DARK)
        c.rect(0, 0, self.w, self.h, fill=1)

        c.setFillColor(C_WHITE)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(20, 70, "CareerIQ Report")

        c.setFont("Helvetica", 8)
        c.setFillColor(C_TEXT2)
        c.drawString(20, 55, f"Report ID: {self.pred_id}")

        c.drawRightString(self.w - 20, 55, self.ts)


# ─── SECTION TITLE ──────────────────────────────────────
class SectionTitle(SafeFlowable):
    def __init__(self, w, text):
        self.w = w
        self.h = 26
        self.text = text

    def draw(self):
        c = self.canv
        c.setFillColor(C_CARD)
        c.roundRect(0, 0, self.w, 26, 6, fill=1)

        c.setFillColor(C_WHITE)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(12, 8, self.text)


# ─── SCORE BAR (FIXED) ─────────────────────────────────
class ScoreBar(SafeFlowable):
    def __init__(self, w, label, score):
        self.w = float(w)
        self.h = 18
        self.label = str(label)
        self.score = max(0, min(100, int(score)))

    def draw(self):
        c = self.canv

        c.setFont("Helvetica", 8)
        c.setFillColor(C_TEXT2)
        c.drawString(0, 6, self.label)

        bx = 100
        bw = self.w - 140

        c.setFillColor(C_CARD2)
        c.rect(bx, 5, bw, 8, fill=1)

        c.setFillColor(C_TEAL)
        c.rect(bx, 5, bw * self.score / 100, 8, fill=1)

        c.setFillColor(C_WHITE)
        c.drawRightString(self.w, 6, f"{self.score}%")


# ─── CONFIDENCE ─────────────────────────────────────────
class ConfidenceMeter(SafeFlowable):
    def __init__(self, confidence):
        self.w = 80
        self.h = 80
        self.confidence = confidence

    def draw(self):
        c = self.canv
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(C_PURP2)
        c.drawCentredString(40, 40, f"{int(self.confidence)}%")

        c.setFont("Helvetica", 8)
        c.setFillColor(C_TEXT3)
        c.drawCentredString(40, 25, "MATCH")


# ─── ROADMAP STEP (FIXED) ───────────────────────────────
class RoadmapStep(SafeFlowable):
    def __init__(self, w, title, desc, idx):
        self.w = w
        self.h = 40
        self.title = str(title)
        self.desc = str(desc)[:70]
        self.idx = idx

    def draw(self):
        c = self.canv

        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(C_WHITE)
        c.drawString(10, 25, self.title)

        c.setFont("Helvetica", 8)
        c.setFillColor(C_TEXT2)
        c.drawString(10, 10, self.desc)


# ─── SAFE STACK ─────────────────────────────────────────
def vstack(items, w):
    clean = []
    for i in items:
        clean.append(Paragraph(str(i), ParagraphStyle(
            "x", fontSize=8, textColor=C_TEXT2
        )))
    return Table([[x] for x in clean], colWidths=[w])


# ─── MAIN FUNCTION ──────────────────────────────────────
def generate_career_pdf(result, form_data, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=A4)

    story = []

    ts = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")

    # ─ Header
    story.append(HeaderBanner(500, result.get("prediction_id", 1), ts))
    story.append(Spacer(1, 10))

    # ─ Career
    story.append(SectionTitle(500, "Recommended Career"))
    story.append(Paragraph(result.get("title", "N/A"), ParagraphStyle("a")))

    # ─ Skills (SAFE LOOP)
    story.append(SectionTitle(500, "Skills"))
    skills = result.get("skills", [])[:8]

    for i, s in enumerate(skills):
        story.append(ScoreBar(500, s, 70))

    # ─ Roadmap (LIMITED FIX)
    story.append(SectionTitle(500, "Roadmap"))
    roadmap = result.get("roadmap", [])[:5]

    for i, r in enumerate(roadmap):
        story.append(RoadmapStep(500, r.get("title","Step"), r.get("desc",""), i))

    # ─ Decision Tree SAFE LIMIT
    story.append(SectionTitle(500, "Decision Path"))
    for i, d in enumerate(result.get("decision_path", [])[:6]):
        story.append(Paragraph(str(d.get("rule","")), ParagraphStyle("b")))

    # ─ Footer
    story.append(HRFlowable(width=500))
    story.append(Paragraph("CareerIQ AI System", ParagraphStyle("f")))

    doc.build(story)

    return output_path