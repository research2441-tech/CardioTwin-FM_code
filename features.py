"""Hand-crafted landmark features (last observation carried forward) for static baselines,
and per-record feature vectors for the recurrent Dynamic-DeepHit baseline."""
import numpy as np


def record_features(C):
    """Per-record engineered features (N,R,F) + availability masks; F identical across records."""
    parts = []
    for m, sp in C["mods"].items():
        X, M = sp["X"], sp["mask"]
        if sp["kind"] == "wave":
            Lx = X.shape[-1]
            w = X.reshape(X.shape[:-1] + (10, Lx // 10)).mean(-1)              # N,R,C,10
            f = np.concatenate([w, X.max(-1, keepdims=True), X.min(-1, keepdims=True)], -1)
            f = f.reshape(X.shape[:2] + (-1,))
        elif sp["kind"] == "seq":
            f = np.concatenate([X.mean(-1), X.std(-1), X.max(-1)], -1)
        else:
            f = X
        parts.append(f * M[..., None])
        parts.append(M[..., None].astype(np.float32))
    if "wear_steps" in C:
        # exposure/treatment vector recorded at visits (standardised), so that every model sees the same exposures
        parts.append(((C["u"] - C["u_mu"]) / C["u_sd"] * C["u_mask"][..., None]).astype(np.float32))
    return np.concatenate(parts, -1).astype(np.float32)


def landmark_features(C, rows, L, RF=None):
    """LOCF of each modality's engineered features up to landmark L, plus time since last observation."""
    RF = record_features(C) if RF is None else RF
    t = C["t"][rows]
    valid = C["valid"][rows] & (t <= L + 1e-6)
    out = []
    col = 0
    for m, sp in C["mods"].items():
        X = sp["X"]
        if sp["kind"] == "wave":
            nf = X.shape[2] * 12
        elif sp["kind"] == "seq":
            nf = X.shape[2] * 3
        else:
            nf = X.shape[-1]
        M = sp["mask"][rows] & valid
        last = np.where(M.any(1), (np.arange(t.shape[1])[None] * M).max(1), 0)
        f = RF[rows, last, col:col + nf] * M.any(1)[:, None]
        since = np.where(M.any(1), L - t[np.arange(len(rows)), last], 5.0)
        out += [f, M.any(1)[:, None].astype(np.float32), since[:, None]]
        col += nf + 1
    nvis = (valid & C["visit"][rows]).sum(1, keepdims=True).astype(np.float32)
    if "wear_steps" in C:
        um = C["u_mask"][rows] & valid
        last = np.where(um.any(1), (np.arange(t.shape[1])[None] * um).max(1), 0)
        ul = (C["u"][rows, last] - C["u_mu"]) / C["u_sd"] * um.any(1)[:, None]
        out.append(nvis)
        return np.concatenate(out + [ul], 1).astype(np.float32)
    return np.concatenate(out + [nvis], 1).astype(np.float32)
