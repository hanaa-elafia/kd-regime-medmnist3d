"""Figure 1: study protocol. Every fact comes from the frozen protocol (fiche de gel v1)
and the B2a addendum (addendum_B2a_v1.md). No result is shown."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_PHASE = "#E8EEF6"   # phases
C_DOC = "#FFF4DC"     # written-before documents
C_TEST = "#FBE3DA"    # protected test
EDGE = "#3A3A3A"

fig, ax = plt.subplots(figsize=(6.2, 4.9))
ax.set_xlim(0, 100); ax.set_ylim(-14, 62); ax.axis("off")

def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=color, ec=EDGE, lw=0.9))
    ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=8.8, weight="bold")
    ax.text(x + w / 2, y + h - 7.0, body, ha="center", va="top", fontsize=7.6, linespacing=1.3)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=EDGE))

# top row: A -> freeze -> B
box(0.5, 34, 30, 27, "Phase A", "screening and pilot:\nmodels, augmentation,\nbounds of T and \u03b1\n(validation data only)", C_PHASE)
box(37, 38, 24, 19, "Frozen protocol", "rules, \u03b4 = 0.02,\nverdicts, seeds\n(Suppl. File S1)", C_DOC)
box(67.5, 34, 32, 27, "Phase B (main)", "student alone and\n8 KD schedules\n27 conditions, seeds 0\u20134\npivot: seeds 0\u201314\n2,043 runs", C_PHASE)
arrow(31.5, 47.5, 36, 47.5); arrow(62, 47.5, 66.5, 47.5)

# bottom row: B2a <- addendum <- analysis
box(67.5, 0, 32, 26, "Phase B analysis", "run once on the\nCSV files: paired\ngains, 95% intervals,\nHolm, verdicts", C_PHASE)
box(37, 2, 24, 22, "B2a addendum", "hypotheses\nHB1\u2013HB3, class\nrules, written\nbefore any run\n(Suppl. File S2)", C_DOC)
box(0.5, 0, 30, 26, "Phase B2a", "confirmatory:\nstudent alone vs\nvanilla KD, 27 conditions,\nnew seeds 15\u201324,\n360 runs", C_PHASE)
arrow(83.5, 33, 83.5, 27); arrow(66.5, 13, 62, 13); arrow(36, 13, 31.5, 13)

# protected test band
ax.add_patch(FancyBboxPatch((0.5, -13), 99, 9, boxstyle="round,pad=0.3,rounding_size=1",
                            fc=C_TEST, ec="#B5523A", lw=0.8, ls="--"))
ax.text(50, -8.5, "Protected test set (all phases): evaluated once per retained run,\n"
        "after checkpoint selection on validation; never used for any choice",
        ha="center", va="center", fontsize=7.6, color="#7A2E1C")

fig.tight_layout()
fig.savefig("figures/Figure_1_protocol.pdf")
fig.savefig("figures/Figure_1_protocol.png", dpi=300)
