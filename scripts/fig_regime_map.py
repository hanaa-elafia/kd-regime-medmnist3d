"""Figure 2: regime map. One point per condition (27).
x = test macro-F1 of the student alone (Phase B, Table 5).
Grey points: Phase B vanilla KD gain (5 seeds, exploratory).
Coloured points: Phase B2a vanilla KD gain (10 new seeds, confirmatory);
filled = class A (established gain), open = not established (class B or unclassified).
Values from outputs/analysis/ (Phase B) and outputs/analysis_b2a/b2a_conditions.csv (B2a).
Before submission: read these values from the CSV files instead of the dicts below."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

student = {"OrganMNIST3D": [0.909, 0.944, 0.953],
           "AdrenalMNIST3D": [0.668, 0.709, 0.715],
           "FractureMNIST3D": [0.440, 0.470, 0.488]}
# [teacher][student]: res_w16_b1, res_w32_b1, res_w32_b2 x S1, S2, S3
gainB = {"OrganMNIST3D": [[-0.002, 0.003, 0.003], [-0.004, 0.004, 0.008], [-0.003, 0.001, 0.010]],
         "AdrenalMNIST3D": [[0.010, -0.011, 0.007], [0.029, 0.002, -0.002], [0.029, -0.018, -0.026]],
         "FractureMNIST3D": [[0.057, 0.087, 0.062], [0.097, 0.086, 0.077], [0.057, 0.055, 0.074]]}
gainB2a = {"OrganMNIST3D": [[-0.002, 0.003, 0.012], [-0.002, 0.003, 0.013], [0.004, -0.010, 0.009]],
           "AdrenalMNIST3D": [[0.019, -0.016, -0.001], [0.026, -0.011, 0.017], [0.012, -0.029, -0.014]],
           "FractureMNIST3D": [[0.076, 0.053, 0.050], [0.097, 0.073, 0.043], [0.081, 0.061, 0.041]]}
classA = {"OrganMNIST3D": [[0, 0, 1], [0, 0, 1], [0, 0, 0]],
          "AdrenalMNIST3D": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
          "FractureMNIST3D": [[1, 1, 1], [1, 1, 0], [1, 1, 0]]}
style = {"FractureMNIST3D": ("#D55E00", "o", "R1: KD useful (FractureMNIST3D)"),
         "AdrenalMNIST3D": ("#0072B2", "s", "R2: uncertain (AdrenalMNIST3D)"),
         "OrganMNIST3D": ("#009E73", "^", "R3: no practical gain (OrganMNIST3D)")}
# small horizontal offsets so the three teachers of one student do not overlap
off = [-0.006, 0.0, 0.006]

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.axhspan(-0.02, 0.02, color="0.9", zorder=0)
ax.axhline(0, color="0.4", lw=0.8, zorder=1)
for ds, (c, m, lab) in style.items():
    first = True
    for t in range(3):
        for s in range(3):
            x = student[ds][s] + off[t]
            ax.scatter(x, gainB[ds][t][s], s=14, c="0.65", marker=m, zorder=2, lw=0)
            filled = classA[ds][t][s] == 1
            ax.scatter(x, gainB2a[ds][t][s], s=48, marker=m, zorder=3,
                       facecolors=c if filled else "white", edgecolors=c, linewidths=1.3,
                       label=lab if first else None)
            first = False
ax.scatter([], [], s=14, c="0.65", marker="o", lw=0, label="Phase B (5 seeds, exploratory)")
ax.scatter([], [], s=48, facecolors="0.3", edgecolors="0.3", marker="o", label="Phase B2a: established gain (class A)")
ax.scatter([], [], s=48, facecolors="white", edgecolors="0.3", linewidths=1.3, marker="o", label="Phase B2a: not established")
ax.fill_between([], [], color="0.9", label=r"equivalence margin $\pm\delta$ = 0.02")
ax.set_xlabel("Test macro-F1 of the student alone")
ax.set_ylabel("Vanilla KD gain in test macro-F1")
ax.set_xlim(0.40, 1.0)
ax.legend(frameon=False, fontsize=7.2, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/Figure_2_regime_map.pdf")
fig.savefig("figures/Figure_2_regime_map.png", dpi=300)
