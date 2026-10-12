"""Fig. 1: CardioTwin-FM architecture (grayscale, print-safe)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.2, "mathtext.fontset": "dejavuserif"})
fig, ax = plt.subplots(figsize=(7.16, 4.25), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")

def box(x, y, w, h, text, fc="white", lw=0.8, ls="-", fs=7.0, bold=False, ec="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", linespacing=1.32)
    return (x, y, w, h)

def arrow(p0, p1, ls="-", lw=0.9, color="black", style="-|>", rad=0.0, ms=7):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, ls=ls,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)

def panel(x, y, w, h, title):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#f2f2f2", ec="#555555", lw=0.7)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="center", fontsize=6.4, weight="bold")

# ---------------------------------------------------------------- column 1: asynchronous inputs
panel(0.5, 4.0, 16.5, 52.5, "Inputs")
mods = ["EHR and labs\n(visits)", "ECG beat\n(2 leads)", "PCG cycle", "Echo indices", "Wearable\n(14-day window)"]
ys = [44.5, 36.0, 27.5, 19.0, 10.5]
for yy, mtxt in zip(ys, mods):
    box(1.6, yy, 14.3, 6.0, mtxt, fs=6.0)
ax.text(8.75, 6.2, "irregular times,\nany subset", ha="center", va="center", fontsize=6.3, style="italic")

# ---------------------------------------------------------------- column 2: stage 1 encoders
panel(19.0, 4.0, 17.5, 52.5, "Stage 1: encoders")
for yy in ys:
    box(20.2, yy, 15.1, 6.0, "encoder $f_m$\ntoken $\\mathbf{e}_{ik}^{m}$", fs=6.0)
    arrow((15.9 + 0.35, yy + 3.0), (20.2 - 0.35, yy + 3.0))
ax.text(27.75, 6.6, "masked InfoNCE\n+ reconstruction", ha="center", va="center", fontsize=6.2, style="italic")

# ---------------------------------------------------------------- column 3: stage 2 assimilation
panel(38.5, 4.0, 19.5, 52.5, "Stage 2: assimilation")
pr = box(40.0, 40.5, 16.5, 9.0, "Predict step\n$\\boldsymbol{\\mu}\\leftarrow\\boldsymbol{\\mu}+\\Delta t\\,\\mathbf{g}(\\boldsymbol{\\mu},\\mathbf{u})$\n$\\mathbf{P}\\leftarrow\\mathbf{P}+\\Delta t\\,\\mathbf{q}$", fs=6.0)
up = box(40.0, 25.5, 16.5, 10.0, "Update step\n$\\boldsymbol{\\Lambda}=\\mathbf{P}^{-1}+\\Sigma_{m}\\boldsymbol{\\pi}^{m}$\npresent tokens only", fs=6.0)
po = box(40.0, 11.5, 16.5, 8.0, "Posterior state\n$\\mathcal{N}(\\boldsymbol{\\mu}_{L},\\mathbf{P}_{L})$\nblocks A|D|R|H", fs=6.0)
# encoders -> update (merge line)
for yy in ys:
    ax.plot([35.3 + 0.35, 37.4], [yy + 3.0, yy + 3.0], color="black", lw=0.7)
ax.plot([37.4, 37.4], [ys[-1] + 3.0, ys[0] + 3.0], color="black", lw=0.7)
arrow((37.4, 30.5), (40.0 - 0.35, 30.5))
# predict -> update -> (loop back) ; update -> posterior
arrow((46.0, 40.5 - 0.35), (46.0, 35.5 + 0.35))
arrow((52.0, 35.5 + 0.35), (52.0, 40.5 - 0.35))
ax.text(52.8, 38.0, "next\nrecord", fontsize=5.4, ha="left", va="center")
arrow((48.25, 25.5 - 0.35), (48.25, 19.5 + 0.35))

# ---------------------------------------------------------------- column 4: stage 3 structural dynamics
panel(60.0, 4.0, 19.5, 52.5, "Stage 3: roll-out")
ex = box(61.3, 42.0, 17.0, 8.0, "Exposures $\\mathbf{u}$\nSBP, LDL, smoking,\nsteps; signed $\\mathbf{B}$", fs=6.0)
ro = box(61.3, 27.0, 17.0, 9.5, "Monte Carlo\nroll-out ($S$ paths)\nblock-DAG drift $\\mathbf{W}$", fs=6.0)
cf = box(61.3, 11.5, 17.0, 9.0, "Counterfactual\n$do(\\mathbf{u}=\\mathbf{u}')$ from $L$\nshared noise", fs=6.0, ls=(0, (3, 2)))
arrow((56.5 + 0.35, 15.5), (61.3 - 0.35, 31.0), rad=0.0)
arrow((69.8, 42.0 - 0.35), (69.8, 36.5 + 0.35))
# exposures also drive the predict step of the filter (carried-forward u between records)
arrow((61.3 - 0.35, 45.0), (56.5 + 0.35, 45.0))
ax.text(58.9, 45.9, "$\\mathbf{u}$", fontsize=6.0, ha="center", va="bottom")
arrow((66.0, 27.0 - 0.35), (66.0, 20.5 + 0.35), ls=(0, (3, 2)))
ax.text(66.6, 23.7, "$\\mathbf{u}\\rightarrow\\mathbf{u}'$", fontsize=6.0, ha="left", va="center")

# ---------------------------------------------------------------- column 5: stage 4 & 5 outputs
panel(81.5, 4.0, 18.0, 52.5, "Stages 4-5: heads")
hz = box(82.6, 40.0, 15.8, 9.5, "Hazard head\n(soft monotone)\nCIF at 1, 3, 5 y", fs=6.0)
cs = box(82.6, 25.0, 15.8, 9.5, "Weighted\nMondrian conformal\nsets $\\{0\\},\\{1\\},\\{0,1\\}$", fs=5.7)
dr = box(82.6, 11.5, 15.8, 8.0, "Risk difference\n$\\Delta\\hat{F}(\\tau)$", fs=6.0)
arrow((78.3 + 0.35, 31.75), (82.6 - 0.35, 44.75))
arrow((90.5, 40.0 - 0.35), (90.5, 34.5 + 0.35))
arrow((78.3 + 0.35, 16.0), (82.6 - 0.35, 15.5), ls=(0, (3, 2)))

# ---------------------------------------------------------------- dashed reverse-mode gradient path (training only)
G = "#555555"
ax.plot([95.0, 95.0], [49.5 + 0.35, 52.0], lw=0.8, ls=(0, (4, 2)), color=G)
ax.plot([95.0, 54.0], [52.0, 52.0], lw=0.8, ls=(0, (4, 2)), color=G)
arrow((54.0, 52.0), (54.0, 49.5 + 0.35), ls=(0, (4, 2)), color=G, lw=0.8)
# legend strip
ax.plot([2, 6], [1.6, 1.6], color="black", lw=0.9); ax.text(6.6, 1.6, "inference data flow", va="center", fontsize=6.0)
ax.plot([27, 31], [1.6, 1.6], color="black", lw=0.9, ls=(0, (3, 2))); ax.text(31.6, 1.6, "counterfactual branch", va="center", fontsize=6.0)
ax.plot([54, 58], [1.6, 1.6], color=G, lw=0.8, ls=(0, (4, 2)))
ax.text(58.6, 1.6, "reverse-mode gradient of the training loss $\\mathcal{L}$ (training only)", va="center", fontsize=6.0)
fig.savefig("../paper/figs/fig1_architecture.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("ok")
