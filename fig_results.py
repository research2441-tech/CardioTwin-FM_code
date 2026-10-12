"""Result figures (grayscale, print-safe) from results.json and saved predictions."""
import json, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "dejavuserif"})
R = json.load(open("../runs/results.json"))
FIG = "../paper/figs"
NAMES = {"cox": "Cox", "rsf": "RSF", "deepsurv": "DeepSurv", "deephit": "DeepHit", "ddh": "Dyn-DeepHit", "mmt": "MMT",
         "twin": "CardioTwin-FM"}
ORDER = ["cox", "rsf", "deepsurv", "deephit", "ddh", "mmt", "twin"]
MK = {"cox": "o", "rsf": "v", "deepsurv": "s", "deephit": "D", "ddh": "^", "mmt": "P", "twin": "*"}
GR = {"cox": "0.55", "rsf": "0.65", "deepsurv": "0.35", "deephit": "0.45", "ddh": "0.25", "mmt": "0.15", "twin": "0.0"}
LS = {"cox": (0, (1, 1)), "rsf": (0, (3, 1, 1, 1)), "deepsurv": (0, (4, 2)), "deephit": (0, (2, 2)), "ddh": (0, (6, 2)),
      "mmt": (0, (5, 1, 1, 1)), "twin": "-"}


def fig_discrimination():
    S = R["sim"]
    sets = [("Int.", None), ("D", "D"), ("E", "E"), ("F", "F")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), dpi=300, gridspec_kw=dict(wspace=0.38))
    models = [m for m in ORDER if m in S["main"]]
    xs = np.arange(len(sets))
    w = 0.11
    for ai, (metric, lab) in enumerate((("C@1", "$C_{1}$"), ("C@5", "$C_{5}$"), ("IBS", "IBS"))):
        ax = axes[ai]
        for k, m in enumerate(models):
            vals, errs = [], []
            for nm, site in sets:
                v = S["main"][m][metric] if site is None else S["ext"][m][site][metric]
                vals.append(v[0]); errs.append(v[1])
            off = (k - (len(models) - 1) / 2) * w
            ax.errorbar(xs + off, vals, yerr=errs, fmt=MK[m], ms=4.2 if m != "twin" else 6.5, color=GR[m], mfc=GR[m] if m == "twin" else "white",
                        mec=GR[m], mew=0.8, elinewidth=0.6, capsize=1.2, label=NAMES[m])
        ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in sets])
        ax.set_xlabel("Test set (internal; external sites D\u2013F)", fontsize=6.6)
        ax.set_ylabel(lab + (" (lower is better)" if metric == "IBS" else ""))
        ax.grid(axis="y", lw=0.3, color="0.85")
        ax.set_title(f"({'abc'[ai]})", fontsize=7.5, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.08), fontsize=6.8, handletextpad=0.2, columnspacing=1.0)
    fig.savefig(f"{FIG}/fig3_discrimination.png", bbox_inches="tight", pad_inches=0.04)


def fig_missing():
    S = R["sim"]
    sc = [("no_ecg", "no\nECG"), ("no_pcg", "no\nPCG"), ("no_echo", "no\necho"), ("no_wear", "no\nwearable"),
          ("ehr_only", "EHR\nonly"), ("stale12", "last year\nwithheld")]
    models = [m for m in ["cox", "deepsurv", "ddh", "mmt", "twin"] if m in S["miss"]]
    fig, ax = plt.subplots(figsize=(3.45, 2.45), dpi=300)
    xs = np.arange(len(sc)); w = 0.15
    for k, m in enumerate(models):
        v = [S["miss"][m][s]["dC@5"][0] for s, _ in sc]; e = [S["miss"][m][s]["dC@5"][1] for s, _ in sc]
        off = (k - (len(models) - 1) / 2) * w
        ax.errorbar(xs + off, v, yerr=e, fmt=MK[m], ms=4 if m != "twin" else 6, color=GR[m], mfc=GR[m] if m == "twin" else "white",
                    mec=GR[m], mew=0.8, elinewidth=0.6, capsize=1.2, label=NAMES[m])
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([s[1] for s in sc], fontsize=6.4)
    ax.set_ylabel("Change in $C_{5}$ vs. complete history")
    ax.grid(axis="y", lw=0.3, color="0.85")
    ax.legend(ncol=3, loc="lower center", fontsize=6.3, handletextpad=0.2, columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
    lo = min(S["miss"][m][s]["dC@5"][0] - S["miss"][m][s]["dC@5"][1] for m in models for s, _ in sc)
    ax.set_ylim(lo - 0.035, 0.012)
    fig.savefig(f"{FIG}/fig4_missing.png", bbox_inches="tight", pad_inches=0.04)


def fig_cf(seed=0):
    sys.path.insert(0, ".")
    from cohort import sim_cohort
    from run_sim import row_sets
    C = sim_cohort("../runs/cardiosim.npz"); S_ = row_sets(C)
    cfrows = np.concatenate([S_["dev"]] + [S_[nm] for nm in C["site_names"][3:]])
    pidx = {p: i for i, p in enumerate(C["policies"])}
    true = C["cf"][cfrows, pidx["sbp_m10"], 2] - C["cf"][cfrows, 0, 2]
    fig = plt.figure(figsize=(7.16, 2.3), dpi=300)
    gs = fig.add_gridspec(1, 4, wspace=0.5, width_ratios=[1, 1, 1, 1.35])
    for k, m in enumerate(["cox", "mmt", "twin"]):
        ax = fig.add_subplot(gs[0, k])
        D = np.load(f"../runs/sim/{m}_s{seed}.npz")
        pred = D["CF_sbp_m10"][:, 19]
        ax.scatter(100 * true, 100 * pred, s=1.2, color="0.25", alpha=0.35, lw=0, rasterized=True)
        lim = [100 * true.min() - 0.5, 1.0]
        ax.plot(lim, lim, color="0.0", lw=0.7, ls=(0, (3, 2)))
        ax.set_xlim(lim); ax.set_ylim(-6.5, 1.5)
        ax.set_xlabel("True $\\Delta F_{5}$ (pp)")
        if k == 0:
            ax.set_ylabel("Predicted $\\Delta\\hat{F}_{5}$ (pp)")
        rho = R["sim"]["cf"][m]["sbp_m10@19"]["rho"][0]
        ax.set_title(f"({'abc'[k]}) {NAMES[m] if m != 'twin' else 'Twin'}, $\\rho$={rho:.2f}", fontsize=7, loc="left")
        ax.grid(lw=0.3, color="0.88")
    ax = fig.add_subplot(gs[0, 3])
    pols = [("sbp_m10", "SBP\n−10"), ("ldl_m39", "LDL\n−1"), ("nosmoke", "quit\nsmoking"), ("steps_p2k", "+2k\nsteps")]
    models = [m for m in ["cox", "deepsurv", "ddh", "mmt", "twin"] if m in R["sim"]["cf"]]
    hatches = {"cox": "", "deepsurv": "////", "ddh": "....", "mmt": "xxxx", "twin": ""}
    fills = {"cox": "white", "deepsurv": "white", "ddh": "white", "mmt": "white", "twin": "0.15"}
    xs = np.arange(len(pols)); w = 0.16
    for k, m in enumerate(models):
        v = [R["sim"]["cf"][m][f"{p}@19"]["rho"][0] for p, _ in pols]
        e = [R["sim"]["cf"][m][f"{p}@19"]["rho"][1] for p, _ in pols]
        off = (k - (len(models) - 1) / 2) * w
        ax.bar(xs + off, v, w * 0.92, yerr=e, color=fills[m], edgecolor="0.0", hatch=hatches[m], lw=0.5,
               error_kw=dict(lw=0.5, capsize=1), label=NAMES[m])
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([p[1] for p in pols], fontsize=5.5, rotation=0)
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title("(d)", fontsize=7, loc="left")
    ymin = min(0, min(R["sim"]["cf"][m][f"{p}@19"]["rho"][0] for m in models for p, _ in pols)) - 0.1
    ax.set_ylim(ymin, 1.05)
    ax.legend(fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17), handletextpad=0.3, columnspacing=0.8)
    fig.savefig(f"{FIG}/fig5_counterfactual.png", bbox_inches="tight", pad_inches=0.04)


def fig_conformal():
    cf = R["sim"]["conf"]
    sets = ["dev", "D", "E", "F"]; lab = {"dev": "Internal", "D": "Site D", "E": "Site E", "F": "Site F"}
    modes = [("naive", "Split (complete case)"), ("ipcw", "IPCW marginal"), ("ipcw_label", "IPCW + label"),
             ("ipcw_label_avail", "IPCW + label × avail."), ("shift", "+ shift weights")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), dpi=300, gridspec_kw=dict(wspace=0.35))
    hat = ["", "////", "....", "", "xxxx"]; fc = ["white", "white", "white", "0.25", "white"]
    xs = np.arange(len(sets)); w = 0.16
    for ai, (key, ylab) in enumerate((("cov1", "Coverage, patients with event"), ("cov0", "Coverage, event-free"), ("ambig", "Ambiguous sets {0,1}"))):
        ax = axes[ai]
        for k, (mode, ml) in enumerate(modes):
            v, e, xx = [], [], []
            for si, s in enumerate(sets):
                kk = f"{s}|{mode}|5"
                if kk in cf:
                    v.append(cf[kk][key][0]); e.append(cf[kk][key][1]); xx.append(si)
            off = (k - (len(modes) - 1) / 2) * w
            ax.bar(np.array(xx) + off, v, w * 0.92, yerr=e, color=fc[k], edgecolor="0.0", hatch=hat[k], lw=0.5,
                   error_kw=dict(lw=0.5, capsize=1), label=ml)
        if key != "ambig":
            ax.axhline(0.9, color="0.0", lw=0.7, ls=(0, (3, 2)))
            ax.set_ylim(0, 1.08)
        ax.set_xticks(xs); ax.set_xticklabels([lab[s] for s in sets], fontsize=6.5)
        ax.set_ylabel(ylab); ax.set_title(f"({'abc'[ai]})", fontsize=7.5, loc="left")
        ax.grid(axis="y", lw=0.3, color="0.88")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.09), fontsize=6.5, handletextpad=0.3, columnspacing=1.0)
    fig.savefig(f"{FIG}/fig6_conformal.png", bbox_inches="tight", pad_inches=0.04)


def fig_calib_dca():
    S = R["sim"]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85), dpi=300, gridspec_kw=dict(wspace=0.45))
    ax = axes[0]
    for m in ["cox", "deepsurv", "mmt", "twin"]:
        c = S["calcurve"][m]
        ax.plot(c["pred"], c["obs"], marker=MK[m], ms=3 if m != "twin" else 4.5, color=GR[m], ls=LS[m], lw=0.9,
                mfc="white" if m != "twin" else GR[m], mew=0.7, label=NAMES[m])
    mx = max(max(S["calcurve"][m]["pred"] + S["calcurve"][m]["obs"]) for m in ["cox", "deepsurv", "mmt", "twin"]) * 1.05
    ax.plot([0, mx], [0, mx], color="0.6", lw=0.6)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("Predicted 5-y risk", fontsize=6.5); ax.set_ylabel("Observed (KM)", fontsize=6.5)
    ax.set_title("(a)", fontsize=7, loc="left"); ax.tick_params(labelsize=6)
    ax = axes[1]
    thr = np.array(S["dca"]["thr"])
    for m in ["cox", "deepsurv", "mmt", "twin"]:
        ax.plot(thr, S["dca"][m], color=GR[m], ls=LS[m], lw=0.9, label=NAMES[m])
    ax.plot(thr, S["dca"]["all"], color="0.6", lw=0.7, ls=(0, (1, 2)), label="Treat all")
    ax.axhline(0, color="0.3", lw=0.5)
    ax.set_ylim(-0.02, max(max(S["dca"][m]) for m in ["cox", "twin"]) * 1.25)
    ax.set_xlabel("Threshold probability", fontsize=6.5); ax.set_ylabel("Net benefit", fontsize=6.5)
    ax.set_title("(b)", fontsize=7, loc="left"); ax.tick_params(labelsize=6)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.15), fontsize=6, handletextpad=0.3, columnspacing=0.8)
    fig.savefig(f"{FIG}/fig7_calibration_dca.png", bbox_inches="tight", pad_inches=0.04)


def fig_real():
    FR = R["Framingham"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), dpi=300, gridspec_kw=dict(wspace=0.3, width_ratios=[1, 1.25]))
    ax = axes[0]
    Ls = ["L0", "L6.7", "L12.7"]
    for m in ["cox", "deepsurv", "ddh", "mmt", "twin"]:
        if m not in FR["models"]:
            continue
        v = [FR["models"][m][L]["C@10"][0] for L in Ls]
        ax.plot([0, 6.7, 12.7], v, marker=MK[m], color=GR[m], ls=LS[m], ms=3.5 if m != "twin" else 5.5,
                mfc="white" if m != "twin" else GR[m], mew=0.7, lw=0.9, label=NAMES[m])
    ax.set_xticks([0, 6.7, 12.7]); ax.set_xticklabels(["0 (exam 1)", "6.7 (exam 2)", "12.7 (exam 3)"])
    ax.set_xlabel("Landmark (years)"); ax.set_ylabel("$C_{10}$ (10-year horizon)")
    ax.set_title("(a) Framingham: discrimination by landmark", fontsize=7, loc="left")
    ax.grid(lw=0.3, color="0.88"); ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax = axes[1]
    pols = [("sbp_m10", "SBP −10 mmHg", 0.80, "[@ettehad2016bp]"), ("tc_m39", "Cholesterol −1 mmol/L", 0.78, "[@ctt2010statin]")]
    models = [m for m in ["cox", "deepsurv", "ddh", "mmt", "twin"] if m in FR["models"]]
    ys = []
    yy = 0
    for p, pl, ref, _ in pols:
        for m in models:
            v = FR["models"][m][f"CF_{p}"]["RR_long"]
            ax.errorbar(v[0], yy, xerr=v[1], fmt=MK[m], color=GR[m], ms=4 if m != "twin" else 6,
                        mfc="white" if m != "twin" else GR[m], mew=0.7, elinewidth=0.6, capsize=1.2)
            ys.append((yy, NAMES[m]))
            yy -= 1
        ax.plot([ref, ref], [yy + 0.5, yy + len(models) + 0.5], color="0.0", lw=0.8, ls=(0, (3, 2)))
        ax.text(ref - 0.006, yy + 0.75, f"trial\nRR {ref:.2f}", ha="right", va="center", fontsize=5.8)
        ax.text(0.565, yy + len(models) + 0.65, pl, ha="left", fontsize=6.3, style="italic")
        yy -= 1.4
    ax.set_yticks([y for y, _ in ys]); ax.set_yticklabels([n for _, n in ys], fontsize=6.3)
    ax.axvline(1.0, color="0.5", lw=0.5)
    ax.set_xlabel("Population risk ratio at 10 years, intervention vs. factual")
    ax.set_title("(b) Framingham: counterfactual effect vs. trial evidence", fontsize=7, loc="left")
    ax.set_xlim(0.55, 1.14)
    ax.grid(axis="x", lw=0.3, color="0.88")
    fig.savefig(f"{FIG}/fig8_framingham.png", bbox_inches="tight", pad_inches=0.04)


if __name__ == "__main__":
    which = sys.argv[1:] or ["disc", "miss", "cf", "conf", "cal", "real"]
    for w in which:
        {"disc": fig_discrimination, "miss": fig_missing, "cf": fig_cf, "conf": fig_conformal, "cal": fig_calib_dca,
         "real": fig_real}[w]()
    print("figs ok")
