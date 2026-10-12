"""Censoring-weighted, availability-stratified (Mondrian) split conformal prediction sets for
horizon-specific event status, with optional covariate-shift weights for a new site."""
import numpy as np
from metrics import km


def horizon_labels(Trel, E, tau):
    """y=1 event by tau, y=0 event-free beyond tau, -1 unknown (censored before tau)."""
    y = np.full(len(Trel), -1, int)
    y[(E > 0) & (Trel <= tau)] = 1
    y[Trel > tau] = 0
    return y


def ipcw(Trel, E, tau, Gtime, Gevent):
    """Inverse probability of remaining uncensored up to min(T, tau), censoring KM from (Gtime, Gevent)."""
    at = np.minimum(Trel, tau) - 1e-6
    G = np.clip(km(Gtime, 1 - Gevent, at), 0.05, 1.0)
    return 1.0 / G


def wquantile(scores, w, w_test, level):
    o = np.argsort(scores)
    s, ww = scores[o], w[o]
    cum = np.cumsum(ww) / (ww.sum() + w_test)
    k = np.searchsorted(cum, level, side="left")
    return np.inf if k >= len(s) else s[k]


def availability_group(C, rows, L, window=1.0):
    """Modalities seen in the year before the landmark. 0: imaging-rich (echo or PCG), 1: ECG with EHR,
    2: EHR and/or wearable only."""
    t = C["t"][rows]
    v = C["valid"][rows] & (t <= L + 1e-6) & (t > L - window)
    seen = {m: (C["mods"][m]["mask"][rows] & v).any(1) for m in C["mods"]}
    g = np.full(len(rows), 2)
    if "ecg" in seen:
        g[seen["ecg"]] = 1
    rich = np.zeros(len(rows), bool)
    for m in ("echo", "pcg"):
        if m in seen:
            rich |= seen[m]
    g[rich] = 0
    return g


class ConformalSets:
    """mode: 'naive' (complete-case, marginal), 'ipcw' (marginal), 'ipcw_label' (label-Mondrian),
    'ipcw_label_avail' (label x availability Mondrian)."""
    def __init__(self, alpha=0.1, mode="ipcw_label_avail"):
        self.alpha, self.mode = alpha, mode

    def _strata(self, y, g):
        if self.mode in ("naive", "ipcw"):
            return np.zeros_like(y)
        if self.mode == "ipcw_label":
            return y
        return y * 10 + g

    def fit(self, p, Trel, E, tau, g, Gtime, Gevent, shift_w=None):
        y = horizon_labels(Trel, E, tau)
        k = y >= 0
        w = np.ones(len(y)) if self.mode == "naive" else ipcw(Trel, E, tau, Gtime, Gevent)
        if shift_w is not None:
            w = w * shift_w
        s = np.where(y == 1, 1 - p, p)            # nonconformity = 1 - p_hat(y)
        st = self._strata(np.where(k, y, 0), g)
        self.cal = dict(s=s[k], w=w[k], st=st[k])
        self.tau = tau
        return self

    def qhat(self, stratum, w_test):
        c = self.cal
        m = c["st"] == stratum
        if m.sum() < 30:              # fall back to the label stratum when a stratum is too small
            m = (c["st"] // 10 == stratum // 10) if self.mode == "ipcw_label_avail" else np.ones_like(m)
        return wquantile(c["s"][m], c["w"][m], w_test if w_test is not None else c["w"][m].mean(), 1 - self.alpha)

    def predict(self, p, g, w_test=None):
        """Returns boolean (n,2) set membership for labels {0,1}."""
        n = len(p)
        out = np.zeros((n, 2), bool)
        for lab in (0, 1):
            s = p if lab == 0 else 1 - p
            for i in range(n):
                wt = None if w_test is None else w_test[i]
                st = self._strata(np.array([lab]), np.array([g[i]]))[0]
                out[i, lab] = s[i] <= self.qhat(st, wt)
        return out


def set_metrics(S, Trel, E, tau, Gtime, Gevent):
    y = horizon_labels(Trel, E, tau)
    k = y >= 0
    w = ipcw(Trel, E, tau, Gtime, Gevent)
    cov = S[np.arange(len(y)), np.clip(y, 0, 1)]
    r = {}
    r["cov"] = float((w[k] * cov[k]).sum() / w[k].sum())
    for lab in (0, 1):
        m = k & (y == lab)
        r[f"cov{lab}"] = float((w[m] * cov[m]).sum() / max(w[m].sum(), 1e-9)) if m.any() else np.nan
    size = S.sum(1)
    r["size"] = float(size.mean())
    r["single"] = float((size == 1).mean())
    r["ambig"] = float((size == 2).mean())
    r["empty"] = float((size == 0).mean())
    return r


def domain_weights(Xs, Xt, Xq, seed=0, clip=(0.05, 20.0)):
    """Density-ratio weights p_t(x)/p_s(x) from a logistic domain classifier (source Xs vs target Xt),
    evaluated at Xq."""
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([Xs, Xt]); d = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.05, max_iter=2000).fit((X - mu) / sd, d)
    pr = clf.predict_proba((Xq - mu) / sd)[:, 1]
    w = pr / (1 - pr) * len(Xs) / len(Xt)
    return np.clip(w, *clip)
