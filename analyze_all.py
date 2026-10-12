"""Produce every number used in the manuscript (results.json) from saved predictions."""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from metrics import evaluate, bootstrap_diff, bh, net_benefit, km_calibration, km
from conformal import ConformalSets, set_metrics, availability_group, domain_weights, horizon_labels

OUT = "../runs/results.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(res, open(OUT, "w"), indent=1, default=float)


def msd(v):
    v = np.array(v, float)
    return [float(np.nanmean(v)), float(np.nanstd(v))]


# ============================================================================ simulation
def sim_part():
    from cohort import sim_cohort
    from run_sim import row_sets, POLS, SCEN, RUNS
    from twin import at_risk
    C = sim_cohort("../runs/cardiosim.npz")
    S = row_sets(C)
    grid = C["grid_dt"] * np.arange(1, C["K"] + 1)
    trL = at_risk(C, np.where(C["split"] == 0)[0], 3.0)
    yt, ye = C["T"][trL] - 3.0, C["E"][trL]
    HZ = C["horizons"]

    def load(m, seeds=(0, 1, 2, 3, 4)):
        out = {}
        for s in seeds:
            f = f"{RUNS}/{m}_s{s}.npz"
            if os.path.exists(f) and os.path.getsize(f) > 0:
                out[s] = dict(np.load(f))
        return out

    def ev(F, rows, L=3.0):
        return evaluate(yt, ye, C["T"][rows] - L, C["E"][rows], F, grid, HZ)

    models = ["cox", "rsf", "deepsurv", "deephit", "ddh", "mmt", "twin"]
    sim = {"n": {k: int(len(v)) for k, v in S.items()}}
    sim["events5"] = {k: int(((C["T"][v] - 3.0 <= 5) & (C["E"][v] > 0)).sum()) for k, v in S.items() if k in ("dev", "cal", "D", "E", "F")}
    # ---- main table, external sites, dynamic landmarks, missing scenarios
    main, ext, dyn, miss, cost = {}, {}, {}, {}, {}
    avgF = {}
    for m in models:
        R = load(m)
        if not R:
            continue
        sd = list(R)
        vals = [ev(R[s]["F_dev"], S["dev"]) for s in sd]
        main[m] = {k: msd([v[k] for v in vals]) for k in vals[0]}
        main[m]["seeds"] = len(sd)
        avgF[m] = np.mean([R[s]["F_dev"] for s in sd], 0)
        ext[m] = {}
        for site in ("D", "E", "F"):
            vv = [ev(R[s][f"F_{site}"], S[site]) for s in sd]
            ext[m][site] = {k: msd([v[k] for v in vv]) for k in ("C@1", "C@3", "C@5", "IBS", "ECE@5", "Brier@5")}
        dyn[m] = {}
        for key, L in (("dev_L1", 1.0), ("dev_L2", 2.0), ("dev", 3.0)):
            vv = [ev(R[s][f"F_{key}"], S[key], L) for s in sd]
            dyn[m][str(L)] = {k: msd([v[k] for v in vv]) for k in ("C@1", "C@3", "C@5", "IBS")}
        miss[m] = {}
        for sc in SCEN:
            if sc == "full":
                continue
            vv = [ev(R[s][f"M_{sc}"], S["dev"]) for s in sd]
            base = [ev(R[s]["F_dev"], S["dev"]) for s in sd]
            miss[m][sc] = {"C@5": msd([v["C@5"] for v in vv]), "dC@5": msd([v["C@5"] - b["C@5"] for v, b in zip(vv, base)]),
                           "IBS": msd([v["IBS"] for v in vv]), "dIBS": msd([v["IBS"] - b["IBS"] for v, b in zip(vv, base)])}
        cost[m] = {"train_s": msd([R[s]["train_s"] for s in sd]),
                   "nparams": float(np.mean([R[s].get("nparams", -1) for s in sd])),
                   "latency_ms": msd([R[s]["latency_ms"] for s in sd]) if "latency_ms" in R[sd[0]] else None,
                   "epochs": msd([R[s]["epochs"] for s in sd]) if "epochs" in R[sd[0]] else None}
    sim.update(main=main, ext=ext, dyn=dyn, miss=miss, cost=cost)
    # ---- bootstrap: twin vs each baseline at each horizon on seed-averaged predictions
    boot = {}
    if "twin" in avgF:
        pv = []
        keys = []
        for m in models:
            if m == "twin" or m not in avgF:
                continue
            for tau, j in ((1.0, 3), (3.0, 11), (5.0, 19)):
                d_, lo, hi, p = bootstrap_diff(yt, ye, C["T"][S["dev"]] - 3.0, C["E"][S["dev"]], avgF["twin"][:, j], avgF[m][:, j], tau, B=400, seed=1)
                boot[f"{m}@{tau:g}"] = [d_, lo, hi, p]; pv.append(p); keys.append(f"{m}@{tau:g}")
        q = bh(pv)
        for k, qq in zip(keys, q):
            boot[k].append(float(qq))
    sim["boot"] = boot
    # ---- counterfactual fidelity
    cfrows = np.concatenate([S["dev"]] + [S[nm] for nm in C["site_names"][3:]])
    pidx = {p: i for i, p in enumerate(C["policies"])}
    smokers = C["u"][cfrows][:, :, 2].max(1) > 0
    cf = {}
    abls = ["twin_nostruct", "twin_mono", "twin_nopen", "twin_noipw", "twin_nostale", "twin_lastvisit", "twin_nosample", "twin_nopre"]
    for m in models + abls:
        R = load(m, (0, 1, 2, 3, 4) if m in models else (0, 1, 2))
        if not R:
            continue
        cf[m] = {}
        for pol in POLS:
            for hj, h in ((11, 1), (19, 2)):
                true = C["cf"][cfrows, pidx[pol], h] - C["cf"][cfrows, pidx["factual"], h]
                sel = ~np.isnan(true)
                if pol == "nosmoke":
                    sel &= smokers
                st = []
                for s in R:
                    pred = R[s][f"CF_{pol}"][:, hj]
                    e = pred[sel] - true[sel]
                    st.append([np.sqrt(np.mean(e ** 2)), spearmanr(pred[sel], true[sel])[0], np.mean(pred[sel]),
                               np.mean(np.sign(pred[sel]) == np.sign(true[sel]))])
                st = np.array(st)
                cf[m][f"{pol}@{hj}"] = {"rmse": msd(st[:, 0]), "rho": msd(st[:, 1]), "pred": msd(st[:, 2]), "sign": msd(st[:, 3]),
                                        "true": float(np.mean(true[sel])), "n": int(sel.sum())}
    sim["cf"] = cf
    # ---- ablations (seeds 0-2 for both full and ablated)
    abl = {}
    for m in ["twin"] + abls:
        R = load(m, (0, 1, 2))
        if not R:
            continue
        vals = [ev(R[s]["F_dev"], S["dev"]) for s in R]
        a = {k: msd([v[k] for v in vals]) for k in ("C@1", "C@3", "C@5", "IBS", "ECE@5")}
        ext_c = [np.mean([ev(R[s][f"F_{site}"], S[site])["C@5"] for site in ("D", "E", "F")]) for s in R]
        a["extC@5"] = msd(ext_c)
        a["ehr_only_dC@5"] = msd([ev(R[s]["M_ehr_only"], S["dev"])["C@5"] - v["C@5"] for s, v in zip(R, vals)])
        a["stale_dC@5"] = msd([ev(R[s]["M_stale12"], S["dev"])["C@5"] - v["C@5"] for s, v in zip(R, vals)])
        a["cf_rho_mean"] = msd([np.mean([spearmanr(R[s][f"CF_{p}"][:, 19][(~np.isnan(C["cf"][cfrows, pidx[p], 2])) & (smokers if p == "nosmoke" else True)],
                                                     (C["cf"][cfrows, pidx[p], 2] - C["cf"][cfrows, 0, 2])[(~np.isnan(C["cf"][cfrows, pidx[p], 2])) & (smokers if p == "nosmoke" else True)])[0]
                                          for p in POLS]) for s in R])
        a["seeds"] = len(R)
        abl[m] = a
    sim["abl"] = abl
    # ---- conformal
    conf = {}
    Rt = load("twin")
    if Rt:
        cal = S["cal"]
        Tc, Ec = C["T"][cal] - 3.0, C["E"][cal]
        gcal = availability_group(C, cal, 3.0)
        for tau, j in ((1.0, 3), (3.0, 11), (5.0, 19)):
            for setname in ("dev", "D", "E", "F"):
                rows = S[setname]
                Tt, Et = C["T"][rows] - 3.0, C["E"][rows]
                gt = availability_group(C, rows, 3.0)
                modes = ["naive", "ipcw", "ipcw_label", "ipcw_label_avail"] + (["shift", "local"] if setname != "dev" else [])
                for mode in modes:
                    acc = []
                    for s in Rt:
                        pc, pt = Rt[s]["F_cal"][:, j], Rt[s][f"F_{setname}"][:, j]
                        if mode == "local":
                            # 30% of the site's patients (labelled) recalibrate; coverage measured on the other 70%
                            rr = np.random.default_rng(100 + s).permutation(len(rows))
                            lc, le = rr[:int(0.3 * len(rows))], rr[int(0.3 * len(rows)):]
                            cs = ConformalSets(0.1, "ipcw_label_avail").fit(pt[lc], Tt[lc], Et[lc], tau, gt[lc], Tt[lc], Et[lc])
                            acc.append(set_metrics(cs.predict(pt[le], gt[le]), Tt[le], Et[le], tau, Tt[le], Et[le]))
                            continue
                        if mode == "shift":
                            Xs = np.c_[Rt[s]["mu_cal"], np.eye(3)[gcal]]
                            Xt = np.c_[Rt[s][f"mu_{setname}"], np.eye(3)[gt]]
                            ws = domain_weights(Xs, Xt, Xs, clip=(0.1, 10.0)); wt = domain_weights(Xs, Xt, Xt, clip=(0.1, 10.0))
                            cs = ConformalSets(0.1, "ipcw_label_avail").fit(pc, Tc, Ec, tau, gcal, Tc, Ec, shift_w=ws)
                            Sx = cs.predict(pt, gt, w_test=wt)
                        else:
                            cs = ConformalSets(0.1, mode).fit(pc, Tc, Ec, tau, gcal, Tc, Ec)
                            Sx = cs.predict(pt, gt)
                        acc.append(set_metrics(Sx, Tt, Et, tau, Tt, Et))
                    conf[f"{setname}|{mode}|{tau:g}"] = {k: msd([a[k] for a in acc]) for k in acc[0]}
        # group-conditional coverage of events on dev at 5 y for marginal vs Mondrian
        gc = {}
        rows = S["dev"]; gt = availability_group(C, rows, 3.0)
        for mode in ("ipcw", "ipcw_label_avail"):
            for g in range(3):
                acc = []
                for s in Rt:
                    cs = ConformalSets(0.1, mode).fit(Rt[s]["F_cal"][:, 19], Tc, Ec, 5.0, gcal, Tc, Ec)
                    Sx = cs.predict(Rt[s]["F_dev"][:, 19], gt)
                    m_ = gt == g
                    acc.append(set_metrics(Sx[m_], (C["T"][rows] - 3.0)[m_], C["E"][rows][m_], 5.0, C["T"][rows] - 3.0, C["E"][rows]))
                gc[f"{mode}|g{g}"] = {k: msd([a[k] for a in acc]) for k in acc[0]}
                gc[f"{mode}|g{g}"]["n"] = int((gt == g).sum())
        conf["groups"] = gc
    sim["conf"] = conf
    # ---- calibration curves and decision curves (seed-averaged)
    cal_c, dca = {}, {}
    thr = np.round(np.arange(0.05, 0.41, 0.025), 3)
    Tt, Et = C["T"][S["dev"]] - 3.0, C["E"][S["dev"]]
    for m in avgF:
        e, o, p = km_calibration(Tt, Et, avgF[m][:, 19], 5.0, 10)
        cal_c[m] = {"obs": o.tolist(), "pred": p.tolist(), "ece": e}
        dca[m] = net_benefit(Tt, Et, avgF[m][:, 19], 5.0, thr).tolist()
    ev_rate = 1 - km(Tt, Et, 5.0)[0]
    dca["all"] = [ev_rate - (1 - ev_rate) * t / (1 - t) for t in thr]
    dca["thr"] = thr.tolist()
    sim["calcurve"] = cal_c; sim["dca"] = dca
    # ---- learned structure (seed-averaged)
    if Rt:
        Bs = np.mean([Rt[s]["B"] for s in Rt], 0)
        b = Bs.shape[0] // 4
        sim["B_block"] = [[float(Bs[i * b:(i + 1) * b, e].mean()) for e in range(Bs.shape[1])] for i in range(4)]
        qs = np.mean([Rt[s]["q"] for s in Rt], 0)
        sim["q_block"] = [float(qs[i * b:(i + 1) * b].mean()) for i in range(4)]
        Ws = np.mean([Rt[s]["W"] for s in Rt], 0)
        sim["W_diag_block"] = [float(np.diag(Ws)[i * b:(i + 1) * b].mean()) for i in range(4)]
        # posterior variance by availability group (dev)
        gt = availability_group(C, S["dev"], 3.0)
        Pm = np.mean([Rt[s]["P_dev"].mean(1) for s in Rt], 0)
        sim["P_by_group"] = [float(Pm[gt == g].mean()) for g in range(3)]
        sim["n_by_group"] = [int((gt == g).sum()) for g in range(3)]
    # ---- cohort description
    desc = {}
    for i, nm in enumerate(C["site_names"]):
        rows = np.where(C["site"] == i)[0]
        v = C["valid"][rows]
        desc[nm] = {"n": int(len(rows)), "records": float(v.sum(1).mean()), "visits": float((C["visit"][rows] & v).sum(1).mean()),
                    **{f"p_{m}": float(C["mods"][m]["mask"][rows].any(1).mean()) for m in C["mods"]},
                    "mace": float(C["E"][rows].mean()), "age": float(C["mods"]["ehr"]["X"][rows][:, 0, 0].mean())}
    sim["desc"] = desc
    res["sim"] = sim
    save()


# ============================================================================ real cohorts
def real_part(name, reps=(0,)):
    from cohort import framingham_cohort, heartvalve_cohort
    from run_real import load as ld, set_fold, POL
    C = ld(name)
    grid = C["grid_dt"] * np.arange(1, C["K"] + 1)
    HZ = C["horizons"]
    L0 = C["main_landmark"]
    models = ["cox", "rsf", "deepsurv", "deephit", "ddh", "mmt", "twin", "twin_nostruct", "twin_lastvisit", "twin_mono"]
    out = {"models": {}}
    pooled = {}
    for m in models:
        per_rep = []
        for rep in reps:
            fs = [f"../runs/real/{name}_{m}_f{f}_r{rep}.npz" for f in range(5)]
            if not all(os.path.exists(f) for f in fs):
                continue
            Ds = [dict(np.load(f)) for f in fs]
            per_rep.append(Ds)
        if not per_rep:
            continue
        stats = {}
        for key in [f"L{L:g}" for L in C["landmarks"]]:
            L = float(key[1:])
            vals = []
            for Ds in per_rep:
                rows = np.concatenate([D[f"rows_{key}"] for D in Ds]); F = np.concatenate([D[f"F_{key}"] for D in Ds])
                Tr, Er = C["T"][rows] - L, C["E"][rows]
                vals.append(evaluate(Tr, Er, Tr, Er, F, grid, HZ))
            stats[key] = {k: msd([v[k] for v in vals]) for k in vals[0]}
        # scenarios
        for sc in [k[2:] for k in per_rep[0][0] if k.startswith("M_")]:
            vals = []
            for Ds in per_rep:
                rows = np.concatenate([D[f"rows_L{L0:g}"] for D in Ds])
                F = np.concatenate([D[f"M_{sc}"] for D in Ds]); F0 = np.concatenate([D[f"F_L{L0:g}"] for D in Ds])
                Tr, Er = C["T"][rows] - L0, C["E"][rows]
                a, b = evaluate(Tr, Er, Tr, Er, F, grid, HZ), evaluate(Tr, Er, Tr, Er, F0, grid, HZ)
                vals.append({k: a[k] - b[k] for k in a})
            stats[f"M_{sc}"] = {k: msd([v[k] for v in vals]) for k in vals[0]}
        # counterfactual population effects (relative risk at the longest horizon)
        for pol in POL[name]:
            rr = []
            for Ds in per_rep:
                F0 = np.concatenate([D[f"F_L{L0:g}"] for D in Ds]); dF = np.concatenate([D[f"CF_{pol}"] for D in Ds])
                rr.append([float((F0[:, -1] + dF[:, -1]).mean() / F0[:, -1].mean()), float((F0[:, 9] + dF[:, 9]).mean() / F0[:, 9].mean()),
                           float(np.mean(dF[:, -1] < 0))])
            stats[f"CF_{pol}"] = {"RR_long": msd([r[0] for r in rr]), "RR_mid": msd([r[1] for r in rr]), "frac_neg": msd([r[2] for r in rr])}
        if m == "cox" and "coef" in per_rep[0][0]:
            stats["coef_mean"] = np.mean([D["coef"] for D in per_rep[0]], 0).tolist()
        stats["train_s"] = msd([D["train_s"] for Ds in per_rep for D in Ds])
        out["models"][m] = stats
        pooled[m] = per_rep
    # bootstrap twin vs others at main landmark, longest & middle horizon (repeat 0)
    if "twin" in pooled:
        Ds_t = pooled["twin"][0]
        rows = np.concatenate([D[f"rows_L{L0:g}"] for D in Ds_t]); Ft = np.concatenate([D[f"F_L{L0:g}"] for D in Ds_t])
        Tr, Er = C["T"][rows] - L0, C["E"][rows]
        boot, pv, keys = {}, [], []
        for m in pooled:
            if m.startswith("twin"):
                continue
            Ds = pooled[m][0]
            rows_m = np.concatenate([D[f"rows_L{L0:g}"] for D in Ds]); Fm = np.concatenate([D[f"F_L{L0:g}"] for D in Ds])
            order = {r: i for i, r in enumerate(rows_m)}
            Fm = Fm[[order[r] for r in rows]]
            for tau in HZ[1:]:
                j = int(np.argmin(np.abs(grid - tau)))
                d_, lo, hi, p = bootstrap_diff(Tr, Er, Tr, Er, Ft[:, j], Fm[:, j], tau, B=400, seed=2)
                boot[f"{m}@{tau:g}"] = [d_, lo, hi, p]; pv.append(p); keys.append(f"{m}@{tau:g}")
        for k, q in zip(keys, bh(pv)):
            boot[k].append(float(q))
        out["boot"] = boot
        # conformal on real cohort (per fold calibration -> pooled test)
        confr = {}
        for tau in HZ:
            j = int(np.argmin(np.abs(grid - tau)))
            for mode in ("naive", "ipcw", "ipcw_label_avail"):
                acc = []
                for Ds in pooled["twin"]:
                    Ss, Ts_, Es_ = [], [], []
                    for f, D in enumerate(Ds):
                        cr, tr_ = D["rows_cal"], D[f"rows_L{L0:g}"]
                        Tc, Ec = C["T"][cr] - L0, C["E"][cr]
                        gcal = np.zeros(len(cr), int); gte = np.zeros(len(tr_), int)
                        cs = ConformalSets(0.1, mode).fit(D["F_cal"][:, j], Tc, Ec, tau, gcal, Tc, Ec)
                        Ss.append(cs.predict(D[f"F_L{L0:g}"][:, j], gte)); Ts_.append(C["T"][tr_] - L0); Es_.append(C["E"][tr_])
                    Ss, Ts_, Es_ = np.concatenate(Ss), np.concatenate(Ts_), np.concatenate(Es_)
                    acc.append(set_metrics(Ss, Ts_, Es_, tau, Ts_, Es_))
                confr[f"{mode}|{tau:g}"] = {k: msd([a[k] for a in acc]) for k in acc[0]}
        out["conf"] = confr
        # calibration curve of twin and cox at the middle horizon
        cc = {}
        for m in ("twin", "cox", "deepsurv", "mmt"):
            if m not in pooled:
                continue
            Ds = pooled[m][0]
            rows_m = np.concatenate([D[f"rows_L{L0:g}"] for D in Ds]); Fm = np.concatenate([D[f"F_L{L0:g}"] for D in Ds])
            tau = HZ[-1] if name == "Framingham" else HZ[1]
            j = int(np.argmin(np.abs(grid - tau)))
            e, o, p = km_calibration(C["T"][rows_m] - L0, C["E"][rows_m], Fm[:, j], tau, 10 if name == "Framingham" else 5)
            cc[m] = {"obs": o.tolist(), "pred": p.tolist(), "ece": e, "tau": tau}
        out["calcurve"] = cc
    # cohort description
    t = C["t"]; v = C["valid"]
    out["desc"] = {"n": int(len(C["T"])), "records": float(v.sum(1).mean()), "events": int(C["E"].sum()),
                   "followup_median": float(np.median(C["T"])),
                   **{f"p_{m}": float((C["mods"][m]["mask"]).sum() / v.sum()) for m in C["mods"]},
                   "at_risk": {f"L{L:g}": int(((C["T"] > L) & (v & (t <= L + 1e-6)).any(1)).sum()) for L in C["landmarks"]}}
    res[name] = out
    save()


def sens_part():
    """Counterfactual fidelity as a function of the unmeasured-confounding strength gamma_u."""
    from cohort import sim_cohort
    from run_sim import row_sets, POLS
    out = {}
    for g, data, runs in ((0, "../runs/cardiosim_g0.npz", "../runs/sim_g0"), (1, "../runs/cardiosim.npz", "../runs/sim"),
                          (2, "../runs/cardiosim_g2.npz", "../runs/sim_g2")):
        if not os.path.exists(data):
            continue
        C = sim_cohort(data); S = row_sets(C)
        cfrows = np.concatenate([S["dev"]] + [S[nm] for nm in C["site_names"][3:]])
        pidx = {p: i for i, p in enumerate(C["policies"])}
        smokers = C["u"][cfrows][:, :, 2].max(1) > 0
        for m in ("twin", "cox", "mmt"):
            for pol in POLS:
                true = C["cf"][cfrows, pidx[pol], 2] - C["cf"][cfrows, 0, 2]
                sel = ~np.isnan(true) & (smokers if pol == "nosmoke" else True)
                st = []
                for s in (0, 1):
                    f = f"{runs}/{m}_s{s}.npz"
                    if not os.path.exists(f):
                        continue
                    pred = np.load(f)[f"CF_{pol}"][:, 19]
                    st.append([np.sqrt(np.mean((pred[sel] - true[sel]) ** 2)), spearmanr(pred[sel], true[sel])[0],
                               np.mean(pred[sel]) - np.mean(true[sel])])
                if st:
                    st = np.array(st)
                    out[f"{g}|{m}|{pol}"] = {"rmse": msd(st[:, 0]), "rho": msd(st[:, 1]), "bias": msd(st[:, 2]), "true": float(np.mean(true[sel]))}
    res["sens"] = out
    save()


if __name__ == "__main__":
    what = sys.argv[1:] or ["sim", "Framingham", "HeartValve"]
    if "sim" in what:
        sim_part()
    if "Framingham" in what:
        real_part("Framingham", (0,))
    if "HeartValve" in what:
        real_part("HeartValve", (0, 1, 2))
    if "sens" in what:
        sens_part()
    print("saved")
