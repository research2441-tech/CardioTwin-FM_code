"""CardioSim experiment driver: one (model, seed) job per call; predictions saved for analysis."""
import argparse, os, time, copy, json
import numpy as np
from cohort import sim_cohort
import pretrain, twin
from twin import CardioTwin, Feeder, at_risk
from baselines import StaticData, SkModel, DeepStatic, SeqModel, stacked_landmark_rows, grid_targets
from causal import msm_weights
from features import landmark_features

RUNS = os.environ.get("CARDIO_RUNS", "../runs/sim")
DATA = os.environ.get("CARDIO_DATA", "../runs/cardiosim.npz")
EHR_CONF = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]
SCEN = {"full": [], "no_ecg": ["ecg"], "no_pcg": ["pcg"], "no_echo": ["echo"], "no_wear": ["wear"],
        "ehr_only": ["ecg", "pcg", "echo", "wear"], "stale12": "stale"}
POLS = ["sbp_m10", "ldl_m39", "nosmoke", "steps_p2k"]


def row_sets(C):
    L = C["main_landmark"]
    s = {"dev": at_risk(C, np.where(C["split"] == 3)[0], L), "cal": at_risk(C, np.where(C["split"] == 2)[0], L)}
    for i, nm in enumerate(C["site_names"]):
        if i >= 3:
            s[nm] = at_risk(C, np.where((C["site"] == i) & (C["split"] == 4))[0], L)
    s["dev_L1"] = at_risk(C, np.where(C["split"] == 3)[0], 1.0)
    s["dev_L2"] = at_risk(C, np.where(C["split"] == 3)[0], 2.0)
    return s


def scenario_cohort(C, rows, scen):
    C2 = dict(C)
    C2["mods"] = {m: dict(sp) for m, sp in C["mods"].items()}
    if scen == "stale":
        v = C["valid"].copy()
        v[rows] &= C["t"][rows] <= C["main_landmark"] - 1.0
        C2["valid"] = v
        return C2
    for m in scen:
        mk = C["mods"][m]["mask"].copy(); mk[rows] = False
        C2["mods"][m]["mask"] = mk
    return C2


def policy_spec(C, pol):
    """Standardized exposure shifts for the twin (delta, setmask, setval)."""
    nE = len(C["expo_idx"]); d = np.zeros(nE, np.float32); sm = np.zeros(nE, np.float32); sv = np.zeros(nE, np.float32)
    mu, sd = C["u_mu"][C["expo_idx"]], C["u_sd"][C["expo_idx"]]
    if pol == "sbp_m10": d[0] = -10 / sd[0]
    if pol == "ldl_m39": d[1] = -38.7 / sd[1]
    if pol == "nosmoke": sm[2] = 1; sv[2] = (0 - mu[2]) / sd[2]
    if pol == "steps_p2k": d[3] = 2.0 / sd[3]
    return d, sm, sv


def static_perturb(C, X, pol, layout="locf"):
    """Apply the policy to engineered feature matrices: the EHR block (first) and the exposure block (last six
    columns before the landmark column for LOCF rows; last six columns for per-record rows)."""
    X = X.copy()
    mu, sd = C["ehr_stats"]
    umu, usd = C["u_mu"], C["u_sd"]
    uo = X.shape[1] - (7 if layout == "locf" else 6)          # start of the exposure block
    if pol == "sbp_m10":
        X[:, 2] += -10 / sd[2]; X[:, uo + 0] += -10 / usd[0]
    if pol == "ldl_m39":
        X[:, 4] += -38.7 / sd[4]; X[:, uo + 1] += -38.7 / usd[1]
    if pol == "nosmoke":
        X[:, 7] = (0 - mu[7]) / sd[7]; X[:, uo + 2] = (0 - umu[2]) / usd[2]
    if pol == "steps_p2k":
        X[:, uo + 3] += 2.0 / usd[3]
        off = 0
        for m, sp in C["mods"].items():
            if m == "wear":
                break
            nf = sp["X"].shape[2] * 12 if sp["kind"] == "wave" else (sp["X"].shape[2] * 3 if sp["kind"] == "seq" else sp["X"].shape[-1])
            off += nf + (2 if layout == "locf" else 1)
        flag = off + 9                                         # wearable availability flag follows its 9 features
        X[:, off + 2] += (2.0 / C["wear_stats"][1][2, 0]) * (X[:, flag] > 0)
    return X


def run(model, seed, quick=False):
    os.makedirs(RUNS, exist_ok=True)
    C = sim_cohort(DATA)
    S = row_sets(C)
    tr, va = np.where(C["split"] == 0)[0], np.where(C["split"] == 1)[0]
    grid = C["grid_dt"] * np.arange(1, C["K"] + 1)
    out = {}
    t0 = time.time()
    embp = f"{RUNS}/emb_s{seed}.npy"
    needs_emb = model.startswith("twin") or model == "mmt"
    if needs_emb and not os.path.exists(embp):
        P, hist = pretrain.pretrain(C, seed=seed, epochs=3 if quick else 12, verbose=True)
        np.save(embp, pretrain.embed(P, C)); np.save(f"{RUNS}/pretrain_hist_s{seed}.npy", hist)
        import pickle; pickle.dump(P, open(f"{RUNS}/enc_s{seed}.pkl", "wb"))
    emb = pretrain.augment(np.load(embp), C) if needs_emb else None

    if model.startswith("twin"):
        cfg = {"epochs": 3 if quick else 30}
        abl = model.split("_", 1)[1] if "_" in model else "full"
        cfg.update({"full": {}, "nostale": {"stale": False}, "nostruct": {"struct": False}, "noipw": {"ipw": False},
                    "lastvisit": {"longitudinal": False}, "nosample": {"sample": False, "S": 1},
                    "nopre": {"pretrained": False}, "mono": {"mono": True, "lr": 2e-3}, "nopen": {"nu": 0.0}}[abl])
        enc = None
        if abl == "nopre":
            import jax
            enc = pretrain.init(jax.random.PRNGKey(seed + 99), C)["enc"]
        if abl == "nopre":
            cfg["demb"] = 64
        else:
            cfg["demb"] = emb.shape[-1]
        m = CardioTwin(C, cfg, seed=seed, enc_params=enc)
        F = Feeder(C, emb, m.cfg)
        F.ipw = np.ones(len(C["T"]), np.float32)
        F.ipw[tr] = msm_weights(C, tr, C["main_landmark"], EHR_CONF, [0, 1])
        m.fit(F, tr, va, verbose=quick)
        out["epochs"] = len(m.hist)
        for k, rows in S.items():
            L = 1.0 if k == "dev_L1" else (2.0 if k == "dev_L2" else C["main_landmark"])
            out[f"F_{k}"] = m.predict(F, rows, L)
        cfrows = np.concatenate([S["dev"]] + [S[nm] for nm in C["site_names"][3:]])
        base = m.predict(F, cfrows, 3.0)
        for pol in POLS:
            d, sm, sv = policy_spec(C, pol)
            out[f"CF_{pol}"] = m.predict(F, cfrows, 3.0, d, sm, sv) - base
        for sc, drop in SCEN.items():
            if sc == "full":
                continue
            C2 = scenario_cohort(C, S["dev"], drop if drop != "stale" else "stale")
            F2 = Feeder(C2, emb, m.cfg, emb_stats=F.emb_stats)
            F2.ipw = F.ipw
            m.C = C2
            out[f"M_{sc}"] = m.predict(F2, S["dev"], 3.0)
            m.C = C
        for k in ["dev", "cal"] + C["site_names"][3:]:
            mu, P = m.state(F, S[k], 3.0)
            out[f"mu_{k}"] = mu; out[f"P_{k}"] = P
        import jax
        out["nparams"] = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(m.p))
        # inference latency per patient (batch of 512)
        t1 = time.time(); m.predict(F, S["dev"][:512], 3.0); out["latency_ms"] = (time.time() - t1) / 512 * 1e3
        if abl == "full":
            import pickle
            pickle.dump({"p": jax.device_get(m.p), "cfg": m.cfg, "emb_stats": F.emb_stats, "ipw": F.ipw},
                        open(f"{RUNS}/twinparams_s{seed}.pkl", "wb"))
            out["W"] = np.asarray(m.p["W"]); out["B"] = np.asarray(twin._B(m.p, m.mk, True)); out["q"] = np.asarray(twin.qvec(m.p, m.cfg))

    elif model in ("cox", "rsf", "deepsurv", "deephit"):
        SD = StaticData(C)
        pr, pl = stacked_landmark_rows(C, tr)
        Xtr = SD.X(pr, pl); jtr, etr, ttr = grid_targets(C, pr, pl)
        vr = at_risk(C, va, 3.0); Xva = SD.X(vr, 3.0); jva, eva, tva = grid_targets(C, vr, 3.0)
        if model in ("cox", "rsf"):
            if model == "rsf" and len(pr) > 6000:
                sub = np.random.default_rng(seed).choice(len(pr), 6000, replace=False)
                mdl = SkModel(model, seed).fit(Xtr[sub], ttr[sub], etr[sub])
            else:
                mdl = SkModel(model, seed).fit(Xtr, ttr, etr)
            cif = lambda X: mdl.cif(X, grid)
        else:
            mdl = DeepStatic(C, model, seed).fit(Xtr, jtr, etr, ttr, Xva, jva, eva, tva)
            cif = lambda X: mdl.cif(X, grid)
        for k, rows in S.items():
            L = 1.0 if k == "dev_L1" else (2.0 if k == "dev_L2" else 3.0)
            out[f"F_{k}"] = cif(SD.X(rows, L))
        cfrows = np.concatenate([S["dev"]] + [S[nm] for nm in C["site_names"][3:]])
        Xcf = SD.X(cfrows, 3.0); base = cif(Xcf)
        for pol in POLS:
            out[f"CF_{pol}"] = cif(static_perturb(C, Xcf, pol)) - base
        for sc, drop in SCEN.items():
            if sc == "full":
                continue
            C2 = scenario_cohort(C, S["dev"], drop)
            out[f"M_{sc}"] = cif(StaticData(C2).X(S["dev"], 3.0))
        out["nparams"] = int(Xtr.shape[1]) if model == "cox" else -1

    elif model in ("ddh", "mmt"):
        mdl = SeqModel(C, model, seed, emb=emb).fit(tr, va)
        for k, rows in S.items():
            L = 1.0 if k == "dev_L1" else (2.0 if k == "dev_L2" else 3.0)
            out[f"F_{k}"] = mdl.cif(rows, L)
        cfrows = np.concatenate([S["dev"]] + [S[nm] for nm in C["site_names"][3:]])
        base = mdl.cif(cfrows, 3.0)
        for pol in POLS:
            if model == "mmt":
                d, sm, sv = policy_spec(C, pol)

                def ov(rr, inp, d=d, sm=sm, sv=sv):
                    u = inp["u"].copy(); um = inp["um"] > 0
                    last = np.where(um.any(1), (np.arange(um.shape[1])[None] * um).max(1), 0)
                    ix = np.arange(len(rr))
                    v = u[ix, last, :4]
                    v = np.where(sm > 0, sv, v + d)
                    u[ix, last, :4] = v
                    return u
                out[f"CF_{pol}"] = mdl.cif(cfrows, 3.0, ue_override=ov) - base
            else:
                RF0 = mdl.RF
                RF = RF0.copy()
                t = C["t"][cfrows]; vis = C["visit"][cfrows] & C["valid"][cfrows] & (t <= 3.0)
                last = np.where(vis.any(1), (np.arange(t.shape[1])[None] * vis).max(1), 0)
                RF[cfrows, last] = static_perturb(C, RF[cfrows, last], pol, layout="rf")
                if pol == "steps_p2k":
                    wv = C["wear_mask"][cfrows] & C["valid"][cfrows] & (t <= 3.0)
                    lw = np.where(wv.any(1), (np.arange(t.shape[1])[None] * wv).max(1), 0)
                    has = wv.any(1)
                    RF[cfrows[has], lw[has]] = static_perturb(C, RF[cfrows[has], lw[has]], pol, layout="rf")
                mdl.RF = RF
                out[f"CF_{pol}"] = mdl.cif(cfrows, 3.0) - base
                mdl.RF = RF0
        for sc, drop in SCEN.items():
            if sc == "full":
                continue
            C2 = scenario_cohort(C, S["dev"], drop)
            m2 = copy.copy(mdl); m2.C = C2
            if model == "ddh":
                from features import record_features
                m2.RF = record_features(C2)
            m2.mm = np.stack([C2["mods"][x]["mask"] for x in C2["mods"]], -1).astype(np.float32)
            out[f"M_{sc}"] = m2.cif(S["dev"], 3.0)
        import jax
        out["nparams"] = sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(mdl.p))
        t1 = time.time(); mdl.cif(S["dev"][:512], 3.0); out["latency_ms"] = (time.time() - t1) / 512 * 1e3
    out["train_s"] = time.time() - t0
    np.savez_compressed(f"{RUNS}/{model}_s{seed}.npz", **out)
    print(model, seed, "done in %.0fs" % out["train_s"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(a.model, a.seed, a.quick)
