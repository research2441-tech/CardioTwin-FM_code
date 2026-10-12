"""Fig. 2: filtered twin state of one internal test patient over the 3-year history window, showing
uncertainty growth between records and contraction at records with different modality subsets."""
import pickle, sys
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cohort import sim_cohort
import pretrain, twin
from twin import CardioTwin, Feeder, at_risk

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.2, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "mathtext.fontset": "dejavuserif"})
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
C = sim_cohort("../runs/cardiosim.npz")
emb = pretrain.augment(np.load(f"../runs/sim/emb_s{seed}.npy"), C)
st = pickle.load(open(f"../runs/sim/twinparams_s{seed}.pkl", "rb"))
m = CardioTwin(C, st["cfg"], seed=seed)
m.p = jax.tree_util.tree_map(jnp.asarray, st["p"])
F = Feeder(C, emb, m.cfg, emb_stats=st["emb_stats"])
rows = at_risk(C, np.where(C["split"] == 3)[0], 3.0)
names = list(C["mods"])
mm = np.stack([C["mods"][n]["mask"] for n in names], -1)
# choose a patient with all modality types in the window and several wearable windows
score = []
for r in rows:
    v = C["valid"][r] & (C["t"][r] <= 3.0)
    have = mm[r][v].any(0)
    score.append(have.sum() * 10 + v.sum() + 5 * (C["E"][r] > 0) * (C["T"][r] < 8))
pid = rows[int(np.argmax(score))]
grid = np.linspace(0.0, 3.0, 181)
b = F.batch(np.repeat(pid, len(grid)), grid)
b = {k: jnp.asarray(v) for k, v in b.items()}
mu, P, u, _ = twin.filter_(m.p, m.mk, m.cfg, b["emb"], b["mm"], b["t"], b["valid"], b["ue"], b["um"], b["L"])
mu, P = np.asarray(mu), np.asarray(P)
# risk at 5 y as a function of landmark time on the same grid
Fc = np.array([m.predict(F, np.array([pid]), float(L))[0, 19] for L in grid])   # same noise draws at every time
Z = mu.shape[1]; bs = Z // 4
fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.2), dpi=300, sharex=True, gridspec_kw=dict(hspace=0.12, height_ratios=[1.9, 1]))
ax = axes[0]
styles = [("A", "-", "0.0"), ("D", (0, (4, 2)), "0.3"), ("R", (0, (1, 1)), "0.45"), ("H", (0, (6, 2, 1, 2)), "0.6")]
for i, (nm, ls, col) in enumerate(styles):
    sb = np.sqrt(P[:, i * bs:(i + 1) * bs]).mean(1)
    ax.plot(grid, sb, color=col, ls=ls, lw=1.0, label=f"block {nm}")
ax.set_ylabel("Posterior s.d. (mean over block)")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17), fontsize=6.3, handlelength=2.2, columnspacing=0.9)
lo, hi = ax.get_ylim(); span = hi - lo
t = C["t"][pid]; v = C["valid"][pid] & (t <= 3.0)
mk = {"ehr": ("|", "EHR"), "ecg": ("o", "ECG"), "pcg": ("s", "PCG"), "echo": ("D", "echo"), "wear": ("^", "wearable")}
for j, n in enumerate(names):
    tt = t[v & C["mods"][n]["mask"][pid]]
    yy = lo - span * (0.12 + 0.1 * j)
    if n == "ehr":
        ax.scatter(tt, np.full(len(tt), yy), marker="|", s=22, color="0.0", lw=0.9, clip_on=False)
    else:
        ax.scatter(tt, np.full(len(tt), yy), marker=mk[n][0], s=10, facecolor="white", edgecolor="0.0", lw=0.6, clip_on=False)
    ax.text(3.07, yy, mk[n][1], fontsize=5.8, va="center")
ax.set_ylim(lo - span * 0.62, hi)
ax.set_yticks([y for y in ax.get_yticks() if lo - 1e-9 <= y <= hi])
ax.axhline(lo - span * 0.05, color="0.75", lw=0.4)
ax.grid(axis="y", lw=0.3, color="0.9")
ax2 = axes[1]
ax2.plot(grid, 100 * Fc, color="0.0", lw=1.0)
ax2.set_ylabel("$\\hat{F}_{5}$ (%)")
ax2.set_xlabel("Time since first record (years)")
ax2.grid(lw=0.3, color="0.88")
fig.savefig("../paper/figs/fig2_trajectory.png", bbox_inches="tight", pad_inches=0.04)
print("patient", pid, "event", C["E"][pid], "T", C["T"][pid], "site", C["site"][pid])
