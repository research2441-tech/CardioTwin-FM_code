"""Real-cohort experiments (Framingham teaching cohort, aortic-valve echo cohort): 5-fold patient-level
cross-validation; out-of-fold predictions saved per (cohort, model, fold, repeat)."""
import argparse, os, time, copy
import numpy as np
import pretrain, twin
from cohort import framingham_cohort, heartvalve_cohort
from twin import CardioTwin, Feeder, at_risk
from baselines import StaticData, SkModel, DeepStatic, SeqModel, stacked_landmark_rows, grid_targets
from features import record_features
from causal import msm_weights

OUT = "../runs/real"
POL = {"Framingham": {"sbp_m10": ("SBP", "shift", -10.0), "tc_m39": ("TC", "shift", -38.7), "nosmoke": ("smoking", "set", 0.0)},
       "HeartValve": {}}
SCN = {"Framingham": {"no_labs": ["labs"], "no_vitals": ["vitals"], "stale": "stale"},
       "HeartValve": {"no_grad": ["echo_grad"], "no_struct": ["echo_struct"], "stale": "stale"}}


def load(name):
    if name == "Framingham":
        return framingham_cohort("../data_raw/framingham.rda")
    return heartvalve_cohort("../data_raw/heart.valve.rda")


def set_fold(C, fold, rep):
    rng = np.random.default_rng(1000 * rep + fold)
    N = len(C["T"])
    folds = C["folds"] if rep == 0 else np.random.default_rng(77 + rep).integers(0, 5, N)
    split = np.full(N, 0, np.int8)
    split[folds == fold] = 3
    rest = np.where(folds != fold)[0]
    perm = rng.permutation(rest)
    nv = int(0.125 * len(rest))
    split[perm[:nv]] = 1; split[perm[nv:2 * nv]] = 2
    C["split"] = split
    return C


def offsets(C, layout):
    off, out = 0, {}
    for m, sp in C["mods"].items():
        out[m] = off
        off += sp["X"].shape[-1] + (2 if layout == "locf" else 1)
    return out


def policy_spec(C, pol):
    nm, kind, amt = POL[C["name"]][pol]
    nE = len(C["expo_idx"]); j = C["expo_names"].index(nm)
    d = np.zeros(nE, np.float32); sm = np.zeros(nE, np.float32); sv = np.zeros(nE, np.float32)
    mu, sd = C["u_mu"][C["expo_idx"]], C["u_sd"][C["expo_idx"]]
    if kind == "shift":
        d[j] = amt / sd[j]
    else:
        sm[j] = 1; sv[j] = (amt - mu[j]) / sd[j]
    return d, sm, sv


RAWLOC = {"Framingham": {"SBP": ("vitals", 0), "TC": ("labs", 0), "smoking": ("history", 2)}}


def perturb_rows(C, X, pol, layout):
    """Apply a policy to engineered feature rows (LOCF or per-record layout)."""
    nm, kind, amt = POL[C["name"]][pol]
    m, col = RAWLOC[C["name"]][nm]
    sp = C["mods"][m]
    c = offsets(C, layout)[m] + col
    X = X.copy()
    if kind == "shift":
        X[:, c] += amt / sp["sd"][col]
    else:
        X[:, c] = (amt - sp["mu"][col]) / sp["sd"][col]
    return X


def scenario(C, rows, drop):
    C2 = dict(C); C2["mods"] = {m: dict(sp) for m, sp in C["mods"].items()}
    if drop == "stale":
        v = C["valid"].copy()
        L = C["main_landmark"]
        if C["name"] == "Framingham":
            v[rows, 2] = False                               # most recent examination unavailable
        else:
            v[rows] &= C["t"][rows] <= L - 1.0
        C2["valid"] = v
        return C2
    for m in drop:
        mk = C["mods"][m]["mask"].copy(); mk[rows] = False; C2["mods"][m]["mask"] = mk
    return C2


def run(name, model, fold, rep=0):
    os.makedirs(OUT, exist_ok=True)
    C = set_fold(load(name), fold, rep)
    tr, va, ca, te = (np.where(C["split"] == k)[0] for k in range(4))
    grid = C["grid_dt"] * np.arange(1, C["K"] + 1)
    L0 = C["main_landmark"]
    sets = {f"L{L:g}": at_risk(C, te, L) for L in C["landmarks"]}
    sets["cal"] = at_risk(C, ca, L0)
    out = {f"rows_{k}": v for k, v in sets.items()}
    seed = 10 * rep + fold
    t0 = time.time()
    emb = None
    if model.startswith("twin") or model == "mmt":
        P, _ = pretrain.pretrain(C, seed=seed, epochs=30 if name == "HeartValve" else 15, rows=np.concatenate([tr, va]), verbose=False, batch=128)
        emb = pretrain.augment(pretrain.embed(P, C), C)
    if model.startswith("twin"):
        abl = model.split("_", 1)[1] if "_" in model else "full"
        cfg = {"epochs": 40 if name == "HeartValve" else 30, "demb": emb.shape[-1], "batch": 64 if name == "HeartValve" else 128}
        cfg.update({"full": {}, "nostruct": {"struct": False}, "lastvisit": {"longitudinal": False}, "nostale": {"stale": False},
                    "noipw": {"ipw": False}, "mono": {"mono": True, "lr": 2e-3}, "nopen": {"nu": 0.0}}[abl])
        if name == "HeartValve":
            cfg["ipw"] = False
        m = CardioTwin(C, cfg, seed=seed)
        Fd = Feeder(C, emb, m.cfg)
        if m.cfg["ipw"] and C["treat_idx"]:
            hist_cols = list(range(len(C["mods"]["history"]["cols"])))
            # denominator: current vitals/labs are not in the history block, so use the record's tabular blocks jointly
            Cw = dict(C); Cw["mods"] = dict(C["mods"])
            Xcat = np.concatenate([C["mods"]["vitals"]["X"], C["mods"]["labs"]["X"], C["mods"]["history"]["X"]], -1)
            Cw["mods"]["ehr"] = dict(X=Xcat, mask=C["valid"])
            Cw["u"] = C["u"]
            nv, nl = C["mods"]["vitals"]["X"].shape[-1], C["mods"]["labs"]["X"].shape[-1]
            conf = list(range(nv + nl)) + [nv + nl + k for k in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)]
            Fd.ipw[tr] = msm_weights(Cw, tr, L0, conf, [nv + nl + 0, nv + nl + 1])
        m.fit(Fd, tr, va)
        out["epochs"] = len(m.hist)
        for k, rows in sets.items():
            L = L0 if k == "cal" else float(k[1:])
            out[f"F_{k}"] = m.predict(Fd, rows, L)
        for pol in POL[name]:
            d, sm, sv = policy_spec(C, pol)
            rows = sets[f"L{L0:g}"]
            out[f"CF_{pol}"] = m.predict(Fd, rows, L0, d, sm, sv) - out[f"F_L{L0:g}"]
        for sc, drop in SCN[name].items():
            C2 = scenario(C, sets[f"L{L0:g}"], drop)
            F2 = Feeder(C2, emb, m.cfg, emb_stats=Fd.emb_stats)
            m.C = C2
            out[f"M_{sc}"] = m.predict(F2, sets[f"L{L0:g}"], L0)
            m.C = C
        for k in ("cal", f"L{L0:g}"):
            mu, Pv = m.state(Fd, sets[k], L0); out[f"mu_{k}"] = mu
    elif model in ("cox", "rsf", "deepsurv", "deephit"):
        SD = StaticData(C)
        pr, pl = stacked_landmark_rows(C, tr)
        Xtr = SD.X(pr, pl); jtr, etr, ttr = grid_targets(C, pr, pl)
        vr = at_risk(C, va, L0); Xva = SD.X(vr, L0); jva, eva, tva = grid_targets(C, vr, L0)
        if model in ("cox", "rsf"):
            mdl = SkModel(model, seed).fit(Xtr, ttr, etr)
        else:
            mdl = DeepStatic(C, model, seed, hidden=(64, 32) if name == "HeartValve" else (128, 64)).fit(Xtr, jtr, etr, ttr, Xva, jva, eva, tva)
        cif = lambda X: mdl.cif(X, grid)
        for k, rows in sets.items():
            L = L0 if k == "cal" else float(k[1:])
            out[f"F_{k}"] = cif(SD.X(rows, L))
        rows = sets[f"L{L0:g}"]; Xb = SD.X(rows, L0)
        for pol in POL[name]:
            out[f"CF_{pol}"] = cif(perturb_rows(C, Xb, pol, "locf")) - out[f"F_L{L0:g}"]
        for sc, drop in SCN[name].items():
            out[f"M_{sc}"] = cif(StaticData(scenario(C, rows, drop)).X(rows, L0))
        if model == "cox":
            out["coef"] = mdl.m.coef_ / mdl.sd
    else:
        mdl = SeqModel(C, model, seed, emb=emb).fit(tr, va)
        for k, rows in sets.items():
            L = L0 if k == "cal" else float(k[1:])
            out[f"F_{k}"] = mdl.cif(rows, L)
        rows = sets[f"L{L0:g}"]
        for pol in POL[name]:
            if model == "mmt":
                d, sm, sv = policy_spec(C, pol)

                def ov(rr, inp, d=d, sm=sm, sv=sv):
                    u = inp["u"].copy(); um = inp["um"] > 0
                    last = np.where(um.any(1), (np.arange(um.shape[1])[None] * um).max(1), 0)
                    ix = np.arange(len(rr)); nE = len(d)
                    u[ix, last, :nE] = np.where(sm > 0, sv, u[ix, last, :nE] + d)
                    return u
                out[f"CF_{pol}"] = mdl.cif(rows, L0, ue_override=ov) - out[f"F_L{L0:g}"]
            else:
                RF0 = mdl.RF; RF = RF0.copy()
                t = C["t"][rows]; vis = C["valid"][rows] & (t <= L0 + 1e-6)
                last = np.where(vis.any(1), (np.arange(t.shape[1])[None] * vis).max(1), 0)
                RF[rows, last] = perturb_rows(C, RF[rows, last], pol, "rf")
                mdl.RF = RF
                out[f"CF_{pol}"] = mdl.cif(rows, L0) - out[f"F_L{L0:g}"]
                mdl.RF = RF0
        for sc, drop in SCN[name].items():
            C2 = scenario(C, rows, drop)
            m2 = copy.copy(mdl); m2.C = C2
            if model == "ddh":
                m2.RF = record_features(C2)
            m2.mm = np.stack([C2["mods"][x]["mask"] for x in C2["mods"]], -1).astype(np.float32)
            out[f"M_{sc}"] = m2.cif(rows, L0)
    out["train_s"] = time.time() - t0
    np.savez_compressed(f"{OUT}/{name}_{model}_f{fold}_r{rep}.npz", **out)
    print(name, model, fold, rep, "%.0fs" % out["train_s"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort"); ap.add_argument("--model"); ap.add_argument("--fold", type=int); ap.add_argument("--rep", type=int, default=0)
    a = ap.parse_args()
    run(a.cohort, a.model, a.fold, a.rep)
