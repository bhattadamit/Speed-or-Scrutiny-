import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------- Figure 1: conceptual model ----------
fig, ax = plt.subplots(figsize=(11, 6.4))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.4); ax.axis("off")

def box(x, y, w, h, text, fc, ec="#1f2d3d", fs=10.5, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold")

def arrow(x1, y1, x2, y2, label=None, color="#1f2d3d", ls="-", lx=0, ly=0, rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, ls=ls,
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=9.5, color=color,
                ha="center", va="center", weight="bold",
                bbox=dict(fc="white", ec="none", pad=1.2))

BL, GR, OR, PU, TE = "#1f6fb2", "#2e8b57", "#d9822b", "#6b4c9a", "#0f8b8d"
box(0.3, 4.7, 2.6, 0.95, "Task–assistant fit\n(TAF)", BL)
box(0.3, 2.75, 2.6, 0.95, "Trust in the\nassistant (TRU)", GR)
box(0.3, 0.8, 2.6, 0.95, "Perceived compliance\nrisk (PCR)", OR)

box(7.9, 4.2, 2.8, 0.95, "Use intensity\n(USE)", PU)
box(7.9, 2.3, 2.8, 0.95, "Verification effort\n(VER)", PU)
box(7.9, 0.4, 2.8, 0.95, "Calibrated reliance\n(CR)", "#8b1e3f")

box(4.0, 5.55, 3.0, 0.7, "Task consequentiality ×\nverifiability", TE, fs=9.5)
box(4.0, 0.05, 3.0, 0.7, "Governance clarity (GOV)", TE, fs=9.5)

def lab(x1,y1,x2,y2,t,text,color,dy=0.0):
    ax.text(x1+(x2-x1)*t, y1+(y2-y1)*t+dy, text, fontsize=9.5, color=color, ha="center", va="center",
            weight="bold", bbox=dict(fc="white", ec="none", pad=1.5))
segs=[(2.95,5.2,7.85,4.85,"P1 (+)",BL,0.5,0.0),
      (2.95,3.4,7.85,4.55,"P2a (+)",GR,0.2,0.0),
      (2.95,3.1,7.85,2.9,"P2b (−)",GR,0.2,0.0),
      (2.95,1.45,7.85,4.3,"P3a (−)",OR,0.2,0.0),
      (2.95,1.15,7.85,2.6,"P3b (+)",OR,0.2,0.0)]
for x1,y1,x2,y2,t,c,f,dy in segs:
    arrow(x1,y1,x2,y2,None,c)
for x1,y1,x2,y2,t,c,f,dy in segs:
    lab(x1,y1,x2,y2,f,t,c,dy)
# PCR moderates trust->use
arrow(2.75, 1.8, 4.6, 3.74, None, OR, ls="--", rad=0.25)
ax.text(2.55, 2.2, "P4", fontsize=9.5, color=OR, weight="bold")
arrow(9.3, 4.15, 9.3, 3.3, None, PU)
arrow(9.3, 2.25, 9.3, 1.4, None, "#8b1e3f")
ax.text(9.5, 1.82, "P5", fontsize=9.5, color="#8b1e3f", weight="bold")
arrow(6.4, 5.5, 6.4, 4.98, None, TE, ls="--")
ax.text(6.55, 5.22, "P6", fontsize=9.5, color=TE, weight="bold")
arrow(6.4, 0.8, 6.4, 2.15, None, TE, ls="--")
ax.text(6.55, 1.2, "P7", fontsize=9.5, color=TE, weight="bold")
ax.text(1.6, 0.3, "Institutional scale (P8): boundary\ncondition on PCR and GOV paths",
        ha="center", va="center", fontsize=8.8, style="italic", color="#444")
ax.text(0.3, 6.2, "Solid arrows: direct relationships", fontsize=8.8, color="#444")
ax.text(0.3, 5.95, "Dashed arrows: moderating relationships", fontsize=8.8, color="#444")
plt.tight_layout()
plt.savefig("fig1.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()

# ---------- Figure 2: task-governance matrix ----------
fig, ax = plt.subplots(figsize=(10.5, 6.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
quads = [
    (1.2, 3.4, "#fff3cd", "#b8860b", "DRAFT-AND-VERIFY",
     "Member letters from approved templates\nPolicy and procedure look-up with cited source\nCall-note summaries posted to the member record\nControl: mandatory source check, named reviewer"),
    (5.6, 3.4, "#f8d7da", "#a12a36", "RESTRICT",
     "Adverse-action reason statements\nSuspicious-activity judgements and narratives\nError-resolution (dispute) determinations\nControl: prohibited or second-line sign-off"),
    (1.2, 0.5, "#d4edda", "#1e7e34", "DELEGATE",
     "Internal e-mail and memo drafts\nMeeting and project summaries\nSpreadsheet formulas, reformatting procedures\nControl: sanctioned tool, spot checks"),
    (5.6, 0.5, "#d1ecf1", "#0c5460", "EXPLORE",
     "Process-improvement brainstorming\nTraining scenario generation\nExploratory analysis without member data\nControl: label as unverified, no onward reuse"),
]
for x, y, fc, ec, title, body in quads:
    ax.add_patch(Rectangle((x, y), 4.2, 2.75, fc=fc, ec=ec, lw=1.8))
    ax.text(x + 2.1, y + 2.35, title, ha="center", va="center", fontsize=12.5, weight="bold", color=ec)
    ax.text(x + 2.1, y + 1.15, body, ha="center", va="center", fontsize=9.3, color="#222", linespacing=1.55)
ax.text(3.3, 0.18, "High verifiability (cheap to check)", ha="center", fontsize=10.5, weight="bold")
ax.text(7.7, 0.18, "Low verifiability (costly to check)", ha="center", fontsize=10.5, weight="bold")
ax.text(0.85, 4.78, "High regulatory\nconsequentiality", ha="center", va="center", rotation=90, fontsize=10.5, weight="bold")
ax.text(0.85, 1.88, "Low regulatory\nconsequentiality", ha="center", va="center", rotation=90, fontsize=10.5, weight="bold")
plt.tight_layout()
plt.savefig("fig2.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()

# ---------- Figure 3: published sector indicators ----------
labels = ["Credit unions that have\ndeployed generative AI\n(Cornerstone, 2026)",
          "Banks that have\ndeployed generative AI\n(Cornerstone, 2026)",
          "Credit unions\nimplementing AI\n(Wipfli, 2026)",
          "Credit unions with an\nenterprise-wide AI roadmap\n(Wipfli, 2026)"]
vals = [59, 49, 66, 16]
cols = ["#1f6fb2", "#8fb8de", "#2e8b57", "#d9822b"]
fig, ax = plt.subplots(figsize=(9, 5.2))
bars = ax.bar(range(4), vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", fontsize=12, weight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.8)
ax.set_ylabel("Share of respondents (%)", fontsize=10.5); ax.set_ylim(0, 80)
ax.spines[["top", "right"]].set_visible(False); ax.yaxis.grid(True, color="#e3e3e3"); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("fig3.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
print("ok")
