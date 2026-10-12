"""CardioSim: mechanistic multimodal longitudinal cardiovascular cohort simulator.

Each simulated patient carries a four-block latent cardiac state
  A  atherosclerotic burden, D  myocardial (systolic) dysfunction,
  R  electrical / arrhythmic substrate, H  haemodynamic load,
driven by modifiable exposures u = (SBP, LDL, smoking, daily steps) and by
clinician-initiated treatments (antihypertensive AH, statin ST) whose
initiation depends on the latent state (confounding by indication).
Observations arrive asynchronously: clinic visits (EHR, ECG median beat,
PCG cycle envelope, echo measurements) and interim 14-day wearable windows.
Because the structural equations are known, individual counterfactual risks
under sustained interventions can be computed exactly by Monte Carlo with
common random numbers, which real cohorts cannot provide.
"""
import numpy as np

SITES = {
    # name: n, mean age, visit rate /yr, P(ECG), P(PCG), P(echo), P(wearable owner), ECG device, extra
    "A": dict(n=3000, age=62, vrate=1.3, pecg=.90, ppcg=.50, pecho=.50, pwear=.30, dev="std12", sbp=0, smk=0, treat=1.0),
    "B": dict(n=3000, age=60, vrate=1.2, pecg=.85, ppcg=.45, pecho=.45, pwear=.40, dev="std12", sbp=0, smk=0, treat=1.0),
    "C": dict(n=2600, age=64, vrate=1.4, pecg=.90, ppcg=.40, pecho=.55, pwear=.25, dev="std12b", sbp=2, smk=0, treat=1.0),
    "D": dict(n=1300, age=58, vrate=0.9, pecg=.70, ppcg=.10, pecho=.05, pwear=.20, dev="legacy", sbp=3, smk=.05, treat=.8),
    "E": dict(n=1300, age=50, vrate=0.5, pecg=.50, ppcg=.05, pecho=.10, pwear=.95, dev="std12", sbp=-2, smk=-.05, treat=.9),
    "F": dict(n=1200, age=60, vrate=0.7, pecg=.60, ppcg=.00, pecho=.00, pwear=.05, dev="single", sbp=8, smk=.12, treat=.5),
}
DEV_SITES, EXT_SITES = ["A", "B", "C"], ["D", "E", "F"]
DT = 1.0 / 12.0          # monthly integration step (years)
T_MAX = 10.0             # simulated follow-up (years)
LANDMARK = 3.0           # forecasting origin for the main task (years)
HORIZONS = (1.0, 3.0, 5.0)
N_ECG, N_PCG, N_WD = 100, 100, 14
ECHO_NAMES = ["LVEF", "E/e'", "LVMI", "WMSI", "LAVI"]
EHR_NAMES = ["age", "sex", "SBP", "DBP", "LDL", "HDL", "HbA1c", "smoke", "BMI", "AH", "ST", "logNTproBNP", "eGFR",
             "LDL_obs", "BNP_obs"]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class Params:
    """Structural coefficients (per year). gamma_u scales unmeasured confounding."""
    def __init__(self, gamma_u=1.0):
        self.gamma_u = gamma_u
        # latent drifts
        self.A = dict(c=0.035, sbp=0.040, ldl=0.055, smk=0.11, diab=0.03, act=0.020, age=0.010, b=0.045)
        self.D = dict(c=0.008, sbp=0.018, A=0.030, R=0.015)
        self.R = dict(c=0.015, age=0.012, D=0.020, sbp=0.012, H=0.015)
        self.H = dict(c=0.015, sbp=0.028, bmi=0.010, age=0.012, D=0.020)
        self.sd = dict(A=0.05, D=0.04, R=0.05, H=0.05)
        # hazards (log scale, per year)
        self.mi = dict(c=-5.0, A=0.80, smk=0.45, sbp=0.10)
        self.hf = dict(c=-5.9, D=1.05, H=0.55, af=0.40)
        self.cd = dict(c=-6.7, D=0.95, R=0.50, A=0.35)
        self.ncv = dict(c=-6.2, age=0.075, smk=0.30)
        # treatment effects on exposures
        self.ah_drop, self.st_frac = 12.0, 0.40


def _baseline(rng, site, n, P):
    s = SITES[site]
    age = np.clip(rng.normal(s["age"], 9, n), 35, 85)
    sex = rng.binomial(1, .5, n).astype(float)
    bmi = np.clip(rng.normal(27.5, 4.5, n), 17, 48)
    b = rng.normal(0, 1, n)                       # unmeasured health-behaviour factor
    diab = rng.binomial(1, sig(-2.6 + .12 * (bmi - 27) + .03 * (age - 55) + .3 * b * P.gamma_u)).astype(float)
    smk = rng.binomial(1, np.clip(sig(-1.1 + .6 * b - .02 * (age - 55)) + s["smk"], 0, 1)).astype(float)
    act = np.clip(7.5 - .04 * (age - 55) - .8 * b - .1 * (bmi - 27) + rng.normal(0, 1.5, n), 1, 15)
    sbp_set = 118 + .5 * (age - 50) + .9 * (bmi - 27) + 3 * diab + s["sbp"] + rng.normal(0, 12, n)
    ldl_set = np.clip(125 + .4 * (age - 50) + 6 * b * P.gamma_u + rng.normal(0, 30, n), 50, 260)
    hdl = np.clip(52 - .8 * (bmi - 27) - 4 * sex + rng.normal(0, 10, n), 20, 100)
    egfr = np.clip(100 - .8 * (age - 40) - 5 * diab + rng.normal(0, 12, n), 15, 130)
    A = .04 * (age - 50) + .3 * smk + .006 * (ldl_set - 100) + .01 * (sbp_set - 120) + .3 * diab \
        + .25 * b * P.gamma_u + rng.normal(0, .5, n)
    D = .15 * np.maximum(A, 0) + rng.normal(0, .3, n)
    R = .03 * (age - 55) + rng.normal(0, .4, n)
    H = .02 * (age - 55) + .012 * (sbp_set - 120) + .04 * (bmi - 27) + rng.normal(0, .4, n)
    AH = rng.binomial(1, sig(-1.3 + .05 * (sbp_set - 140) + .4 * np.maximum(A, 0))).astype(float)
    ST = rng.binomial(1, sig(-1.9 + .02 * (ldl_set - 130) + .8 * diab + .4 * np.maximum(A, 0))).astype(float)
    return dict(age0=age, sex=sex, bmi=bmi, b=b, diab=diab, smk=smk, act=act, sbp_set=sbp_set, ldl_set=ldl_set,
                hdl=hdl, egfr=egfr, A=A, D=D, R=R, H=H, AH=AH, ST=ST)


def _step(st, rng, t, P, policy=None, noise=None, treat_scale=1.0, visit_mask=None):
    """Advance all patients one month. Returns per-patient event codes (0 none, 1 MACE, 2 non-CV death).
    policy: dict with optional keys dsbp, dldl, nosmoke, dsteps applied after policy start time.
    noise: pre-drawn standard normals / uniforms (common random numbers)."""
    n = st["A"].shape[0]
    z = noise if noise is not None else dict(e=rng.normal(size=(8, n)), u=rng.random(size=(6, n)))
    age = st["age0"] + t
    # exposures (true, current)
    sbp = st["sbp_set"] + .5 * t - P.ah_drop * st["AH"]
    ldl = st["ldl_set"] * (1 - P.st_frac * st["ST"])
    smk, act = st["smk"].copy(), st["act"].copy()
    if policy is not None:
        sbp = sbp + policy.get("dsbp", 0.0)
        ldl = np.maximum(ldl + policy.get("dldl", 0.0), 30)
        if policy.get("nosmoke", False):
            smk = np.zeros_like(smk)
        act = np.clip(act + policy.get("dsteps", 0.0), 0, 20)
    pa, pd_, pr, ph = P.A, P.D, P.R, P.H
    sq = np.sqrt(DT)
    dA = (pa["c"] + pa["sbp"] * (sbp - 120) / 10 + pa["ldl"] * (ldl - 100) / 40 + pa["smk"] * smk + pa["diab"] * st["diab"]
          - pa["act"] * (act - 7) + pa["age"] * (age - 55) / 10 + pa["b"] * st["b"] * P.gamma_u)
    dD = pd_["c"] + pd_["sbp"] * (sbp - 120) / 10 + pd_["A"] * np.maximum(st["A"], 0) + pd_["R"] * np.maximum(st["R"], 0)
    dR = pr["c"] + pr["age"] * (age - 55) / 10 + pr["D"] * np.maximum(st["D"], 0) + pr["sbp"] * (sbp - 120) / 10 \
        + pr["H"] * np.maximum(st["H"], 0)
    dH = ph["c"] + ph["sbp"] * (sbp - 120) / 10 + ph["bmi"] * (st["bmi"] - 27) / 4 + ph["age"] * (age - 55) / 10 \
        + ph["D"] * np.maximum(st["D"], 0)
    st["A"] = st["A"] + dA * DT + P.sd["A"] * sq * z["e"][0]
    st["D"] = st["D"] + dD * DT + P.sd["D"] * sq * z["e"][1]
    st["R"] = st["R"] + dR * DT + P.sd["R"] * sq * z["e"][2]
    st["H"] = st["H"] + dH * DT + P.sd["H"] * sq * z["e"][3]
    # exposure evolution (natural course): smoking cessation, activity random walk
    quit = (z["u"][0] < 0.06 * DT) & (st["smk"] > 0)
    st["smk"] = np.where(quit, 0.0, st["smk"])
    st["act"] = np.clip(st["act"] + 0.15 * (7.5 - .8 * st["b"] - st["act"]) * DT + 0.5 * sq * z["e"][4], 0.5, 18)
    st["sbp_set"] = st["sbp_set"] + 2.0 * sq * z["e"][5] * 0.5
    # hazards
    af = sig(-3 + 1.2 * st["R"])
    lmi = np.exp(P.mi["c"] + P.mi["A"] * st["A"] + P.mi["smk"] * smk + P.mi["sbp"] * (sbp - 120) / 10)
    lhf = np.exp(P.hf["c"] + P.hf["D"] * st["D"] + P.hf["H"] * st["H"] + P.hf["af"] * af)
    lcd = np.exp(P.cd["c"] + P.cd["D"] * st["D"] + P.cd["R"] * st["R"] + P.cd["A"] * st["A"])
    lnc = np.exp(P.ncv["c"] + P.ncv["age"] * (age - 55) + P.ncv["smk"] * smk)
    pm = 1 - np.exp(-(lmi + lhf + lcd) * DT)
    pn = 1 - np.exp(-lnc * DT)
    ev = np.where(z["u"][1] < pm, 1, np.where(z["u"][2] < pn, 2, 0))
    # clinician treatment initiation at visits (confounding by indication through latent A, D)
    if visit_mask is not None:
        sbp_meas = sbp + 8 * z["e"][6]
        p_ah = sig(-3.4 + .085 * (sbp_meas - 140) + .6 * np.maximum(st["A"], 0) + .5 * np.maximum(st["D"], 0)) * treat_scale
        p_st = sig(-3.2 + .03 * (ldl - 130) + 1.0 * st["diab"] + .8 * np.maximum(st["A"], 0) + .03 * (age - 60)) * treat_scale
        st["AH"] = np.where(visit_mask & (z["u"][3] < p_ah), 1.0, st["AH"])
        st["ST"] = np.where(visit_mask & (z["u"][4] < p_st), 1.0, st["ST"])
    return ev, dict(sbp=sbp, ldl=ldl, smk=smk, act=act, af=af, age=age)


def _ecg(rng, st, ex, dev, n_obs):
    """Two-lead median beat (100 samples over one second)."""
    tt = np.linspace(0, 1, N_ECG)[None, :]
    A, D, R, H = (st[k][:, None] for k in "ADRH")
    af = (rng.random((n_obs, 1)) < ex["af"][:, None]).astype(float)
    g = lambda mu, s, a: a * np.exp(-0.5 * ((tt - mu) / s) ** 2)
    qrs_w = 0.010 * (1 + 0.30 * np.maximum(D, 0))
    ramp = 1.0 + 0.22 * np.maximum(H, 0) + 0.004 * (ex["sbp"][:, None] - 120)
    st_dep = -0.07 * np.maximum(A - 0.8, 0)
    t_amp = 0.32 - 0.14 * np.maximum(A - 0.5, 0)
    qt = 0.66 + 0.02 * np.maximum(D, 0)
    p = g(0.22, 0.02, 0.11) * (1 - af) + af * 0.03 * np.sin(2 * np.pi * 7 * tt + rng.random((n_obs, 1)) * 6)
    stseg = st_dep * ((tt > 0.43) & (tt < 0.58))
    l1 = p + g(0.38, .008, -0.10) + g(0.40, qrs_w, ramp) + g(0.42, .010, -0.22) + stseg + g(qt, .045, t_amp)
    l2 = 0.7 * p + g(0.38, .008, -0.25) + g(0.40, qrs_w * 1.1, 0.6 * ramp) + g(0.43, .012, -0.55 * ramp) \
        + 1.3 * stseg + g(qt, .05, 0.8 * t_amp)
    x = np.stack([l1, l2], 1)
    gain = dict(std12=1.0, std12b=1.06, legacy=0.85, single=1.0)[dev]
    noise = dict(std12=.02, std12b=.025, legacy=.04, single=.05)[dev]
    x = gain * x + noise * rng.normal(size=x.shape) + 0.03 * np.sin(2 * np.pi * rng.random((n_obs, 1, 1)) + 2 * np.pi * 0.3 * tt[:, None])
    if dev == "legacy":       # crude low-pass of an older device
        k = np.ones(5) / 5
        x = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 2, x)
    if dev == "single":
        x[:, 1, :] = 0.0
    return x.astype(np.float32)


def _pcg(rng, st, ex, n_obs):
    tt = np.linspace(0, 1, N_PCG)[None, :]
    D, H = st["D"][:, None], st["H"][:, None]
    g = lambda mu, s, a: a * np.exp(-0.5 * ((tt - mu) / s) ** 2)
    s1 = g(.10, .02, 1.0 - .15 * np.maximum(D, 0))
    s2 = g(.45, .018, .8 + .004 * (ex["sbp"][:, None] - 120))
    s3 = g(.60, .03, .55 * sig(2.2 * (D - 1.3)))
    s4 = g(.92, .025, .45 * sig(2.2 * (H - 1.4)))
    mur = .30 * sig(2 * (H - 2.0)) * ((tt > .14) & (tt < .40)) * (0.6 + 0.4 * rng.random((n_obs, N_PCG)))
    x = s1 + s2 + s3 + s4 + mur + 0.05 * np.abs(rng.normal(size=(n_obs, N_PCG)))
    return x[:, None, :].astype(np.float32)


def _echo(rng, st, ex, n_obs):
    A, D, R, H = (st[k] for k in "ADRH")
    lvef = 63 - 9 * np.maximum(D, 0) + rng.normal(0, 4, n_obs)
    ee = 8 + 2.5 * np.maximum(H, 0) + rng.normal(0, 1.5, n_obs)
    lvmi = 85 + 12 * np.maximum(H, 0) + .3 * (ex["sbp"] - 120) + rng.normal(0, 10, n_obs)
    wmsi = 1 + .15 * np.maximum(A - 1, 0) + .2 * np.maximum(D - .5, 0) + rng.normal(0, .05, n_obs)
    lavi = 28 + 4 * np.maximum(R, 0) + 3 * np.maximum(H, 0) + rng.normal(0, 4, n_obs)
    return np.stack([lvef, ee, lvmi, wmsi, lavi], 1).astype(np.float32)


def _wear(rng, st, ex, n_obs, site):
    D, R = st["D"][:, None], st["R"][:, None]
    days_af = rng.random((n_obs, N_WD)) < ex["af"][:, None] * 0.6
    bias = 2.0 if site == "E" else 0.0
    hr = 64 + 3 * np.maximum(D, 0) + 2 * np.maximum(R, 0) - .3 * (ex["act"][:, None] - 7) + bias + rng.normal(0, 2.5, (n_obs, N_WD)) + 14 * days_af
    hrv = 38 - .4 * (ex["age"][:, None] - 55) - 4 * np.maximum(D, 0) - 2 * np.maximum(R, 0) + rng.normal(0, 4, (n_obs, N_WD)) + 10 * days_af
    steps = np.clip(ex["act"][:, None] + rng.normal(0, 1.6, (n_obs, N_WD)), 0, 25)
    return np.stack([hr, hrv, steps], 1).astype(np.float32)


def _ehr(rng, st, ex, n_obs):
    sbp_m = ex["sbp"] + rng.normal(0, 8, n_obs)
    dbp = 0.55 * sbp_m + 8 + rng.normal(0, 6, n_obs)
    ldl_m = ex["ldl"] + rng.normal(0, 10, n_obs)
    ldl_obs = (rng.random(n_obs) > 0.25).astype(float)
    hba1c = 5.4 + 1.6 * st["diab"] + rng.normal(0, .4, n_obs)
    bnp = 4.0 + .9 * np.maximum(st["D"], 0) + .5 * np.maximum(st["H"], 0) + .02 * (ex["age"] - 55) + rng.normal(0, .5, n_obs)
    bnp_obs = (rng.random(n_obs) > 0.5).astype(float)
    x = np.stack([ex["age"], st["sex"], sbp_m, dbp, ldl_m * ldl_obs, st["hdl"], hba1c, ex["smk"], st["bmi"], st["AH"],
                  st["ST"], bnp * bnp_obs, st["egfr"], ldl_obs, bnp_obs], 1)
    return x.astype(np.float32)


def simulate_site(site, seed, P=None, n=None):
    """Simulate one site. Returns dict with per-record arrays and per-patient outcomes."""
    P = P or Params()
    s = SITES[site]
    n = n or s["n"]
    rng = np.random.default_rng(seed)
    st = _baseline(rng, site, n, P)
    wear_owner = rng.random(n) < s["pwear"]
    next_visit = np.zeros(n)                  # first clinic visit at t=0
    next_wear = rng.uniform(0.1, 0.25, n)
    alive = np.ones(n, bool)
    t_event = np.full(n, np.inf)
    ev_type = np.zeros(n, int)
    cens = np.clip(rng.exponential(25, n), 0.5, None)      # random loss to follow-up
    cens = np.minimum(cens, rng.uniform(7.0, T_MAX, n))    # administrative end of follow-up
    recs = []
    snapshots = {}
    steps = int(round(T_MAX / DT))
    for k in range(steps):
        t = k * DT
        if abs(t - LANDMARK) < DT / 2:
            snapshots["landmark"] = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in st.items()}
        observing = alive & (cens > t) & (t <= LANDMARK + 1e-9)
        visit = observing & (next_visit <= t + 1e-9)
        wear = observing & wear_owner & (next_wear <= t + 1e-9) & ~visit
        ev, ex = _step(st, rng, t, P, visit_mask=visit, treat_scale=s["treat"])
        for kind, mask in (("visit", visit), ("wear", wear)):
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            sub = {kk: v[idx] for kk, v in st.items()}
            exs = {kk: v[idx] for kk, v in ex.items()}
            m = idx.size
            rec = dict(pid=idx, t=np.full(m, t, np.float32), kind=kind)
            if kind == "visit":
                rec["ehr"] = _ehr(rng, sub, exs, m)
                rec["has_ecg"] = rng.random(m) < s["pecg"]
                rec["has_pcg"] = rng.random(m) < s["ppcg"]
                rec["has_echo"] = rng.random(m) < np.clip(s["pecho"] * (1 + 0.8 * (sub["D"] > 0.8)), 0, 1)
                rec["ecg"] = _ecg(rng, sub, exs, s["dev"], m)
                rec["pcg"] = _pcg(rng, sub, exs, m)
                rec["echo"] = _echo(rng, sub, exs, m)
                rec["has_wear"] = wear_owner[idx]
                rec["wear"] = _wear(rng, sub, exs, m, site)
                rec["u"] = np.stack([exs["sbp"] + rng.normal(0, 8, m), exs["ldl"], exs["smk"], exs["act"], sub["AH"], sub["ST"]], 1).astype(np.float32)
            else:
                rec["wear"] = _wear(rng, sub, exs, m, site)
            recs.append(rec)
        rate = s["vrate"] * np.exp(0.3 * np.clip(st["D"], -1, 3))
        next_visit = np.where(visit, t + rng.exponential(1 / rate) + 0.15, next_visit)
        next_wear = np.where(wear | visit, t + rng.uniform(0.2, 0.35, n), next_wear)
        new = alive & (ev > 0) & (t_event == np.inf) & (cens > t)
        t_event = np.where(new, t + DT * rng.random(n), t_event)
        ev_type = np.where(new, ev, ev_type)
        alive = alive & ~new
    T = np.minimum(t_event, cens)
    E = ((t_event <= cens) & (ev_type == 1)).astype(int)        # MACE; non-CV death treated as censoring
    out = dict(site=site, n=n, T=T.astype(np.float32), E=E, ev_type=ev_type, base=st, records=recs,
               landmark_state=snapshots.get("landmark"), wear_owner=wear_owner)
    return out


def counterfactual_risk(state, site, seed, policies, horizons=HORIZONS, M=200, P=None, t0=LANDMARK):
    """Exact (Monte Carlo) individual risk of MACE within each horizon after t0 under sustained policies,
    starting from the true latent state at t0. Common random numbers across policies."""
    P = P or Params()
    n = state["A"].shape[0]
    rng = np.random.default_rng(seed)
    hmax = max(horizons)
    steps = int(round(hmax / DT))
    res = {name: np.zeros((n, len(horizons))) for name in policies}
    for mrep in range(M):
        noises = [dict(e=rng.normal(size=(8, n)), u=rng.random(size=(6, n))) for _ in range(steps)]
        visit_draw = rng.random((steps, n))
        for name, pol in policies.items():
            st = {kk: (v.copy() if isinstance(v, np.ndarray) else v) for kk, v in state.items()}
            alive = np.ones(n, bool)
            tev = np.full(n, np.inf)
            for k in range(steps):
                t = t0 + k * DT
                vm = visit_draw[k] < SITES[site]["vrate"] * DT
                ev, _ = _step(st, None, t, P, policy=pol, noise=noises[k], treat_scale=SITES[site]["treat"], visit_mask=vm)
                new = alive & (ev == 1)
                tev = np.where(new, (k + 1) * DT, tev)
                alive = alive & (ev == 0)
            for j, h in enumerate(horizons):
                res[name][:, j] += (tev <= h + 1e-9)
    return {k: v / M for k, v in res.items()}


POLICIES = {
    "factual": {},
    "sbp_m10": {"dsbp": -10.0},
    "ldl_m39": {"dldl": -38.7},
    "nosmoke": {"nosmoke": True},
    "steps_p2k": {"dsteps": 2.0},
}
