"""Censoring-aware evaluation: Uno C-index, time-dependent AUC, Brier / integrated Brier score,
Kaplan-Meier calibration error, decision-curve net benefit, paired bootstrap tests."""
import numpy as np
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, brier_score, integrated_brier_score
from sksurv.util import Surv


def km(time, event, at):
    """Kaplan-Meier survival evaluated at times `at`."""
    o = np.argsort(time)
    t, e = time[o], event[o].astype(bool)
    uniq = np.unique(t[e])
    S, s = [], 1.0
    surv_t, surv_v = [0.0], [1.0]
    n = len(t)
    for u in uniq:
        at_risk = (t >= u).sum()
        d = ((t == u) & e).sum()
        s *= 1 - d / max(at_risk, 1)
        surv_t.append(u); surv_v.append(s)
    surv_t, surv_v = np.array(surv_t), np.array(surv_v)
    idx = np.searchsorted(surv_t, np.atleast_1d(at), side="right") - 1
    return surv_v[np.clip(idx, 0, None)]


def censor_km(time, event):
    """Kaplan-Meier of the censoring distribution G(t) as a callable."""
    def G(at):
        return np.clip(km(time, 1 - event, at), 1e-3, 1)
    return G


def evaluate(ytr_t, ytr_e, yte_t, yte_e, F, grid, horizons, n_bins=10):
    """F: (n, K) cumulative incidence on `grid` (interval ends). Returns dict of metrics."""
    ytr = Surv.from_arrays(ytr_e > 0, ytr_t)
    yte = Surv.from_arrays(yte_e > 0, yte_t)
    res = {}
    tmax = min(yte_t[yte_e > 0].max(), ytr_t.max()) - 1e-3
    for tau in horizons:
        j = int(np.argmin(np.abs(grid - tau)))
        r = F[:, j]
        res[f"C@{tau:g}"] = concordance_index_ipcw(ytr, yte, r, tau=tau)[0]
        try:
            res[f"AUC@{tau:g}"] = cumulative_dynamic_auc(ytr, yte, r, [tau])[0][0]
        except ValueError:
            res[f"AUC@{tau:g}"] = np.nan
        res[f"Brier@{tau:g}"] = brier_score(ytr, yte, 1 - r, [tau])[1][0]
        res[f"ECE@{tau:g}"] = km_calibration(yte_t, yte_e, r, tau, n_bins)[0]
    sel = grid[(grid >= grid[0]) & (grid < tmax)]
    js = [int(np.argmin(np.abs(grid - g))) for g in sel]
    try:
        res["IBS"] = integrated_brier_score(ytr, yte, 1 - F[:, js], sel)
    except ValueError:
        res["IBS"] = np.nan
    return res


def km_calibration(t, e, r, tau, n_bins=10):
    q = np.quantile(r, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(q, r, side="right") - 1, 0, n_bins - 1)
    obs, pred, w = [], [], []
    for k in range(n_bins):
        m = b == k
        if m.sum() < 5:
            continue
        obs.append(1 - km(t[m], e[m], tau)[0]); pred.append(r[m].mean()); w.append(m.mean())
    obs, pred, w = map(np.array, (obs, pred, w))
    return float((w * np.abs(obs - pred)).sum() / w.sum()), obs, pred


def net_benefit(t, e, r, tau, thresholds):
    """IPCW-free decision-curve net benefit using Kaplan-Meier within the treated group."""
    n = len(t)
    out = []
    for pt in thresholds:
        sel = r >= pt
        if sel.sum() == 0:
            out.append(0.0); continue
        risk_sel = 1 - km(t[sel], e[sel], tau)[0]
        p = sel.mean()
        out.append(risk_sel * p - (1 - risk_sel) * p * pt / (1 - pt))
    return np.array(out)


def bootstrap_diff(ytr_t, ytr_e, yte_t, yte_e, r1, r2, tau, B=500, seed=0):
    """Paired bootstrap of the difference in Uno's C-index (r1 - r2) at horizon tau."""
    rng = np.random.default_rng(seed)
    ytr = Surv.from_arrays(ytr_e > 0, ytr_t)
    n = len(yte_t)
    d = []
    for _ in range(B):
        ix = rng.integers(0, n, n)
        if (yte_e[ix] > 0).sum() < 5:
            continue
        y = Surv.from_arrays(yte_e[ix] > 0, yte_t[ix])
        try:
            c1 = concordance_index_ipcw(ytr, y, r1[ix], tau=tau)[0]
            c2 = concordance_index_ipcw(ytr, y, r2[ix], tau=tau)[0]
            d.append(c1 - c2)
        except ValueError:
            pass
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975)), max(p, 1.0 / len(d))


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q
