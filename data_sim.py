"""Build padded tensors for CardioSim cohorts and the counterfactual ground truth."""
import numpy as np
import argparse
from cardiosim import (SITES, DEV_SITES, EXT_SITES, LANDMARK, HORIZONS, Params, simulate_site,
                       counterfactual_risk, POLICIES, N_ECG, N_PCG, N_WD)

R_MAX = 24
SITE_IDX = {s: i for i, s in enumerate(SITES)}


def assemble(out, r_max=R_MAX):
    n = out["n"]
    per = [[] for _ in range(n)]
    for rec in out["records"]:
        for j, pid in enumerate(rec["pid"]):
            per[pid].append((float(rec["t"][j]), rec, j))
    D = dict(
        t=np.zeros((n, r_max), np.float32), valid=np.zeros((n, r_max), bool), visit=np.zeros((n, r_max), bool),
        ehr=np.zeros((n, r_max, 15), np.float32), ehr_m=np.zeros((n, r_max), bool),
        ecg=np.zeros((n, r_max, 2, N_ECG), np.float16), ecg_m=np.zeros((n, r_max), bool),
        pcg=np.zeros((n, r_max, 1, N_PCG), np.float16), pcg_m=np.zeros((n, r_max), bool),
        echo=np.zeros((n, r_max, 5), np.float32), echo_m=np.zeros((n, r_max), bool),
        wear=np.zeros((n, r_max, 3, N_WD), np.float32), wear_m=np.zeros((n, r_max), bool),
        u=np.zeros((n, r_max, 6), np.float32), u_m=np.zeros((n, r_max), bool))
    for i in range(n):
        lst = sorted(per[i], key=lambda z: z[0])[-r_max:]
        for k, (t, rec, j) in enumerate(lst):
            D["t"][i, k] = t
            D["valid"][i, k] = True
            if rec["kind"] == "visit":
                D["visit"][i, k] = True
                D["ehr"][i, k] = rec["ehr"][j]; D["ehr_m"][i, k] = True
                D["u"][i, k] = rec["u"][j]; D["u_m"][i, k] = True
                if rec["has_ecg"][j]:
                    D["ecg"][i, k] = rec["ecg"][j]; D["ecg_m"][i, k] = True
                if rec["has_pcg"][j]:
                    D["pcg"][i, k] = rec["pcg"][j]; D["pcg_m"][i, k] = True
                if rec["has_echo"][j]:
                    D["echo"][i, k] = rec["echo"][j]; D["echo_m"][i, k] = True
                if rec["has_wear"][j]:
                    D["wear"][i, k] = rec["wear"][j]; D["wear_m"][i, k] = True
            else:
                D["wear"][i, k] = rec["wear"][j]; D["wear_m"][i, k] = True
    return D


def build(seed=2026, gamma_u=1.0, M=200, path="cardiosim.npz", cf=True):
    P = Params(gamma_u=gamma_u)
    rng = np.random.default_rng(seed)
    parts, states = [], []
    for s in SITES:
        o = simulate_site(s, seed + 101 * SITE_IDX[s], P=P)
        D = assemble(o)
        D["site"] = np.full(o["n"], SITE_IDX[s], np.int16)
        D["T"], D["E"] = o["T"], o["E"].astype(np.int8)
        D["ev_type"] = o["ev_type"].astype(np.int8)
        D["Ltrue"] = np.stack([o["landmark_state"][k] for k in "ADRH"], 1).astype(np.float32)
        parts.append(D)
        states.append(o["landmark_state"])
    keys = parts[0].keys()
    data = {k: np.concatenate([p[k] for p in parts], 0) for k in keys}
    n = data["T"].shape[0]
    # patient-level split for development sites: 60 / 10 / 15 / 15 (train / val / cal / test)
    split = np.full(n, 4, np.int8)   # 4 = external
    for s in DEV_SITES:
        idx = np.where(data["site"] == SITE_IDX[s])[0]
        perm = rng.permutation(idx)
        a, b, c = int(.6 * len(idx)), int(.7 * len(idx)), int(.85 * len(idx))
        split[perm[:a]], split[perm[a:b]], split[perm[b:c]], split[perm[c:]] = 0, 1, 2, 3
    data["split"] = split
    # counterfactual truth for at-risk test patients (dev test + external)
    data["cf"] = np.full((n, len(POLICIES), len(HORIZONS)), np.nan, np.float32)
    if cf:
        off = 0
        for si, s in enumerate(SITES):
            ns = parts[si]["T"].shape[0]
            sel = np.where(((split[off:off + ns] == 3) | (split[off:off + ns] == 4)) & (parts[si]["T"] > LANDMARK))[0]
            st = {k: (v[sel] if isinstance(v, np.ndarray) and v.shape[0] == ns else v) for k, v in states[si].items()}
            res = counterfactual_risk(st, s, seed + 7 + si, POLICIES, M=M, P=P)
            for pi, name in enumerate(POLICIES):
                data["cf"][off + sel, pi] = res[name]
            off += ns
            print("cf done", s, len(sel), flush=True)
    data["policies"] = np.array(list(POLICIES))
    np.savez_compressed(path, **data)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gamma_u", type=float, default=1.0)
    ap.add_argument("--M", type=int, default=200)
    ap.add_argument("--out", default="cardiosim.npz")
    a = ap.parse_args()
    build(a.seed, a.gamma_u, a.M, a.out)
