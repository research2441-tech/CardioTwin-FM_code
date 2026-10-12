"""Unified longitudinal multimodal cohort container used by every model.

A cohort holds, for N patients and up to R time-stamped records:
  t[N,R], valid[N,R]                    record times (years) and validity
  mods[m] = dict(X, mask, kind, blocks)  modality tensors, availability masks, encoder type,
                                         and the latent blocks a modality is allowed to inform
  u[N,R,U], u_mask[N,R]                  exposure/treatment vectors observed at visits
  T[N], E[N]                             absolute event/censoring time and event indicator
  split[N], site[N]
plus exposure metadata (names, sign of harmful direction, latent blocks affected).
"""
import numpy as np

BLOCKS = ["A", "D", "R", "H"]


def standardise(X, mask, train_rows, axis_feat=-1, eps=1e-6):
    sel = X[train_rows][mask[train_rows]]
    if X.ndim > 3:   # waveforms: per-channel scalar stats
        mu = sel.astype(np.float64).mean(axis=(0, 2), keepdims=True)[0]
        sd = sel.astype(np.float64).std(axis=(0, 2), keepdims=True)[0] + eps
        return ((X - mu) / sd).astype(np.float32) * mask[..., None, None], (mu, sd)
    mu = np.nanmean(sel, 0)
    sd = np.nanstd(sel, 0) + eps
    Z = np.nan_to_num((X - mu) / sd).astype(np.float32)
    return Z * mask[..., None], (mu, sd)


def sim_cohort(path):
    d = dict(np.load(path, allow_pickle=True))
    tr = d["split"] == 0
    mods = {}
    ehr = d["ehr"].copy()
    # missing labs coded by indicators: standardise only observed values
    ehr_nan = ehr.copy()
    for j, mcol in ((4, 13), (11, 14)):
        ehr_nan[..., j] = np.where(ehr[..., mcol] > 0, ehr[..., j], np.nan)
    ehr_std, ehr_stats = standardise(ehr_nan, d["ehr_m"], tr)
    ehr_std[..., 13:] = ehr[..., 13:] * d["ehr_m"][..., None]
    mods["ehr"] = dict(X=ehr_std, mask=d["ehr_m"], kind="tab", blocks=[1, 1, 1, 1])
    ecg, _ = standardise(d["ecg"].astype(np.float32), d["ecg_m"], tr)
    mods["ecg"] = dict(X=ecg, mask=d["ecg_m"], kind="wave", blocks=[1, 1, 1, 0])
    pcg, _ = standardise(d["pcg"].astype(np.float32), d["pcg_m"], tr)
    mods["pcg"] = dict(X=pcg, mask=d["pcg_m"], kind="wave", blocks=[0, 1, 0, 1])
    echo, _ = standardise(d["echo"], d["echo_m"], tr)
    mods["echo"] = dict(X=echo, mask=d["echo_m"], kind="tab", blocks=[1, 1, 1, 1])
    wear, wear_stats = standardise(d["wear"], d["wear_m"], tr)
    mods["wear"] = dict(X=wear, mask=d["wear_m"], kind="seq", blocks=[0, 1, 1, 0])
    # exposures: SBP, LDL, smoking, steps (+ AH, ST as treatment indicators)
    u = d["u"].copy()
    # carry steps from wearable windows into u (same units, thousand steps per day)
    wsteps = d["wear"][:, :, 2, :].mean(-1)
    u_mu = u[tr][d["u_m"][tr]].mean(0)
    u_sd = u[tr][d["u_m"][tr]].std(0) + 1e-6
    # decoder targets for next-record prediction: SBP, DBP, LDL, HDL, HbA1c, BMI, BNP, eGFR, echo(5), wear means(3)
    ehr_cols = [2, 3, 4, 5, 6, 8, 11, 12]
    tgt = np.concatenate([ehr_std[..., ehr_cols], echo, wear.mean(-1)], -1)
    tmask = np.concatenate([np.repeat(d["ehr_m"][..., None], 8, -1), np.repeat(d["echo_m"][..., None], 5, -1),
                            np.repeat(d["wear_m"][..., None], 3, -1)], -1).astype(np.float32)
    tmask[..., 2] *= ehr[..., 13]
    tmask[..., 6] *= ehr[..., 14]
    C = dict(name="CardioSim", t=d["t"], valid=d["valid"], visit=d["visit"], mods=mods,
             u=u, u_mask=d["u_m"], u_mu=u_mu, u_sd=u_sd, wear_steps=wsteps, wear_mask=d["wear_m"],
             expo_idx=[0, 1, 2, 3], expo_names=["SBP", "LDL", "smoking", "steps"],
             expo_sign=[1, 1, 1, -1],
             expo_blocks=[[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
             treat_idx=[4, 5], treat_names=["AH", "ST"],
             tgt=tgt.astype(np.float32), tmask=tmask,
             T=d["T"], E=d["E"].astype(np.int32), split=d["split"], site=d["site"],
             landmarks=[1.0, 2.0, 3.0], main_landmark=3.0, grid_dt=0.25, K=20, horizons=[1.0, 3.0, 5.0],
             cf=d["cf"], policies=list(d["policies"]), Ltrue=d["Ltrue"],
             site_names=["A", "B", "C", "D", "E", "F"],
             # raw-feature locations of exposures for input-perturbation baselines: (modality, column, sd)
             expo_raw=[("ehr", 2, ehr_stats[1][2]), ("ehr", 4, ehr_stats[1][4]), ("ehr", 7, ehr_stats[1][7]),
                       ("wear", 2, None)], ehr_stats=ehr_stats, wear_stats=wear_stats)
    return C


def _tab_mod(F, cols, recs, N, R, rows_valid, blocks=(1, 1, 1, 1)):
    """Build a tabular modality with per-feature missing indicators from a long dataframe."""
    X = np.full((N, R, len(cols)), np.nan, np.float32)
    for (i, k), row in recs:
        X[i, k] = [row[c] for c in cols]
    present = ~np.isnan(X)
    mask = present.any(-1) & rows_valid
    return X, present, mask


def framingham_cohort(path, seed=2026):
    import pyreadr
    f = pyreadr.read_r(path)["framingham"].sort_values(["RANDID", "PERIOD"]).reset_index(drop=True)
    ids = f.RANDID.unique()
    idx = {r: i for i, r in enumerate(ids)}
    N, R = len(ids), 3
    t = np.zeros((N, R), np.float32); valid = np.zeros((N, R), bool)
    recs = []
    for _, row in f.iterrows():
        i, k = idx[row.RANDID], int(row.PERIOD) - 1
        t[i, k] = row.TIME / 365.25; valid[i, k] = True
        recs.append(((i, k), row))
    groups = {"vitals": ["SYSBP", "DIABP", "HEARTRTE", "BMI"],
              "labs": ["TOTCHOL", "GLUCOSE", "HDLC", "LDLC"],
              "history": ["AGE", "SEX", "CURSMOKE", "CIGPDAY", "DIABETES", "BPMEDS", "PREVCHD", "PREVAP", "PREVMI",
                          "PREVSTRK", "PREVHYP", "educ"]}
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, 5, N)
    mods, raw = {}, {}
    for g, cols in groups.items():
        X, pres, mask = _tab_mod(f, cols, recs, N, R, valid)
        raw[g] = X
        mu = np.nanmean(X[valid], 0); sd = np.nanstd(X[valid], 0) + 1e-6
        Z = np.nan_to_num((X - mu) / sd) * pres
        Xf = np.concatenate([Z, pres.astype(np.float32)], -1) * mask[..., None]
        mods[g] = dict(X=Xf.astype(np.float32), mask=mask, kind="tab", blocks=[1, 1, 1, 1], mu=mu, sd=sd, cols=cols)
    # exposures: SBP, total cholesterol, smoking, BMI ; treatment: BP medication
    u = np.stack([raw["vitals"][..., 0], raw["labs"][..., 0], raw["history"][..., 2], raw["vitals"][..., 3],
                  raw["history"][..., 5]], -1)
    u_mask = valid & ~np.isnan(u[..., :4]).any(-1)
    u = np.nan_to_num(u, nan=0.0).astype(np.float32)
    u_mu = np.array([u[..., j][u_mask].mean() for j in range(5)]); u_sd = np.array([u[..., j][u_mask].std() for j in range(5)]) + 1e-6
    T = (f.groupby("RANDID").TIMECVD.first().loc[ids].values / 365.25).astype(np.float32)
    E = f.groupby("RANDID").CVD.first().loc[ids].values.astype(np.int32)
    tgt = np.concatenate([mods[g]["X"][..., :len(c)] for g, c in groups.items()], -1)
    tmask = np.concatenate([np.repeat(mods[g]["mask"][..., None], len(c), -1) for g, c in groups.items()], -1).astype(np.float32)
    split = np.zeros(N, np.int8)
    return dict(name="Framingham", t=t, valid=valid, visit=valid.copy(), mods=mods, u=u, u_mask=u_mask, u_mu=u_mu, u_sd=u_sd,
                expo_idx=[0, 1, 2, 3], expo_names=["SBP", "TC", "smoking", "BMI"], expo_sign=[1, 1, 1, 1],
                expo_blocks=[[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]], treat_idx=[4], treat_names=["BPMEDS"],
                tgt=tgt.astype(np.float32), tmask=tmask, T=T, E=E, split=split, site=np.zeros(N, np.int16), folds=folds,
                landmarks=[0.0, 6.7, 12.7], main_landmark=12.7, grid_dt=0.5, K=20, horizons=[2.0, 5.0, 10.0],
                sex=raw["history"][:, 0, 1])


def heartvalve_cohort(path, seed=2026):
    import pyreadr
    h = pyreadr.read_r(path)["heart.valve"].sort_values(["num", "time"]).reset_index(drop=True)
    ids = h.num.unique(); idx = {r: i for i, r in enumerate(ids)}
    N = len(ids); R = int(h.groupby("num").size().max())
    t = np.zeros((N, R), np.float32); valid = np.zeros((N, R), bool)
    h["log.creat"] = np.log(h["creat"])
    h["hs_sl"] = (h["hs"].astype(str) == "Stentless valve").astype(float)
    cnt = {}
    recs = []
    for _, row in h.iterrows():
        i = idx[row.num]; k = cnt.get(i, 0); cnt[i] = k + 1
        t[i, k] = row.time; valid[i, k] = True
        recs.append(((i, k), row))
    groups = {"echo_struct": ["log.lvmi", "ef"], "echo_grad": ["log.grad"],
              "clinical": ["age", "sex", "bsa", "lvh", "prenyha", "redo", "size", "con.cabg", "log.creat", "dm", "acei", "lv",
                           "emergenc", "hc", "sten.reg.mix", "hs_sl"]}
    mods = {}
    for g, cols in groups.items():
        X, pres, mask = _tab_mod(h, cols, recs, N, R, valid)
        mu = np.nanmean(X[valid], 0); sd = np.nanstd(X[valid], 0) + 1e-6
        Z = np.nan_to_num((X - mu) / sd) * pres
        Xf = np.concatenate([Z, pres.astype(np.float32)], -1) * mask[..., None]
        mods[g] = dict(X=Xf.astype(np.float32), mask=mask, kind="tab", blocks=[1, 1, 1, 1], cols=cols)
    T = h.groupby("num").fuyrs.first().loc[ids].values.astype(np.float32)
    E = h.groupby("num").status.first().loc[ids].values.astype(np.int32)
    tgt = np.concatenate([mods[g]["X"][..., :len(c)] for g, c in groups.items() if g != "clinical"], -1)
    tmask = np.concatenate([np.repeat(mods[g]["mask"][..., None], len(c), -1) for g, c in groups.items() if g != "clinical"], -1).astype(np.float32)
    u = np.zeros((N, R, 1), np.float32)
    rng = np.random.default_rng(seed)
    return dict(name="HeartValve", t=t, valid=valid, visit=valid.copy(), mods=mods, u=u, u_mask=np.zeros((N, R), bool),
                u_mu=np.zeros(1), u_sd=np.ones(1), expo_idx=[0], expo_names=["none"], expo_sign=[0],
                expo_blocks=[[0, 0, 0, 0]], treat_idx=[], treat_names=[], tgt=tgt, tmask=tmask, T=T, E=E,
                split=np.zeros(N, np.int8), site=np.zeros(N, np.int16), folds=rng.integers(0, 5, N),
                landmarks=[1.0, 2.0, 3.0, 4.0], main_landmark=2.0, grid_dt=0.5, K=10, horizons=[1.0, 3.0, 5.0])
