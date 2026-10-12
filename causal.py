"""Stabilised inverse-probability-of-treatment weights for treatment initiation observed during
the twin's history window (marginal structural model weighting)."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def msm_weights(C, rows, L, conf_cols, base_cols, trunc=(0.01, 0.99)):
    """For each treatment in C['treat_idx'], model initiation at each visit (previously untreated)
    given current covariates (denominator) or baseline covariates only (numerator)."""
    rows = np.asarray(rows)
    X = C["mods"]["ehr"]["X"][rows]                    # standardized clinical record at each visit
    t = C["t"][rows]
    vis = C["visit"][rows] & C["valid"][rows] & (t <= L + 1e-6)
    logw = np.zeros(len(rows))
    for ti in C["treat_idx"]:
        A = C["u"][rows][..., ti]
        prev = np.concatenate([np.zeros((len(rows), 1)), A[:, :-1]], 1)
        # previous *visit* value: forward-fill across non-visit records
        pv = np.zeros(len(rows)); prevv = np.zeros_like(A)
        for r in range(A.shape[1]):
            prevv[:, r] = pv
            pv = np.where(vis[:, r], A[:, r], pv)
        first = np.zeros_like(vis)
        seen = np.zeros(len(rows), bool)
        for r in range(A.shape[1]):
            first[:, r] = vis[:, r] & ~seen
            seen |= vis[:, r]
        elig = vis & ~first & (prevv == 0)
        ii, rr = np.where(elig)
        if len(ii) < 50:
            continue
        y = A[ii, rr]
        Xd, Xn = X[ii, rr][:, conf_cols], X[ii, rr][:, base_cols]
        den = LogisticRegression(max_iter=2000, C=1.0).fit(Xd, y).predict_proba(Xd)[:, 1]
        num = LogisticRegression(max_iter=2000, C=1.0).fit(Xn, y).predict_proba(Xn)[:, 1]
        contrib = np.where(y > 0, np.log(num) - np.log(den), np.log(1 - num) - np.log(1 - den))
        np.add.at(logw, ii, contrib)
    w = np.exp(logw)
    lo, hi = np.quantile(w, trunc)
    w = np.clip(w, lo, hi)
    return (w / w.mean()).astype(np.float32)
