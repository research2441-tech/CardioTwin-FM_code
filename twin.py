"""CardioTwin-FM stage 2-4: longitudinal twin assimilation, structural exposure dynamics,
multi-horizon discrete-time survival forecasting and counterfactual roll-out.

Latent state s in R^Z is split into four blocks (A, D, R, H) of size Z/4.
  * assimilation  : precision-weighted Gaussian fusion of whichever modality tokens are present
  * prediction    : ds/dt = W s + B phi(u) + c, covariance inflated by q*dt (staleness)
  * structure     : modality->block masks, block DAG mask on W, exposure->block masks and
                    harmful-direction sign constraints on B, monotone hazard read-out
  * forecasting   : S Monte Carlo trajectories from N(mu_L, P_L) -> discrete hazards on a grid
  * counterfactual: identical noise, exposures replaced by do(u) from the landmark onward
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from nets import dense_init, dense, mlp_init, mlp, enc_init, enc_apply

BLOCK_EDGES = [("A", "D"), ("R", "D"), ("D", "R"), ("H", "R"), ("D", "H")]   # source -> target
BL = ["A", "D", "R", "H"]

DEFAULT = dict(Z=32, S=8, sub=4, struct=True, stale=True, ipw=True, longitudinal=True, pretrained=True,
               lam_dec=0.5, epochs=30, batch=128, lr=5e-3, wd=1e-4, patience=6, hid=16, sample=True,
               mono=False, nu=1.0, eps_mono=0.5)


def masks(cfg, C):
    Z = cfg["Z"]; b = Z // 4
    names = list(C["mods"])
    obs = np.stack([np.repeat(np.array(C["mods"][m]["blocks"], np.float32), b) for m in names], 0)   # M,Z
    W = np.zeros((Z, Z), np.float32)
    for i in range(4):
        W[i * b:(i + 1) * b, i * b:(i + 1) * b] = 1
    Wx = np.zeros((Z, Z), np.float32)
    for s_, t_ in BLOCK_EDGES:
        i, j = BL.index(t_), BL.index(s_)
        Wx[i * b:(i + 1) * b, j * b:(j + 1) * b] = 1
    nE = len(C["expo_idx"])
    Bm = np.stack([np.repeat(np.array(C["expo_blocks"][e], np.float32), b) for e in range(nE)], 1)  # Z,nE
    sign = np.array(C["expo_sign"], np.float32)
    if not cfg["struct"]:
        obs = np.ones_like(obs); W = np.ones_like(W); Wx = np.zeros_like(Wx); Bm = np.ones_like(Bm)
    return dict(obs=jnp.asarray(obs), Wd=jnp.asarray(W), Wx=jnp.asarray(Wx), B=jnp.asarray(Bm), sign=jnp.asarray(sign))


def init(key, cfg, C, demb=64, enc_params=None):
    Z, K = cfg["Z"], C["K"]
    names = list(C["mods"])
    nE = len(C["expo_idx"])
    ks = jax.random.split(key, 12 + len(names))
    demb = cfg.get("demb", demb)
    if cfg.get("adapter", "linear") == "mlp":
        p = {"adapt": {m: mlp_init(ks[i], [demb, 64, 2 * Z], 0.5) for i, m in enumerate(names)}}
    else:
        p = {"adapt": {m: dense_init(ks[i], demb, 2 * Z, 0.5) for i, m in enumerate(names)}}
    o = len(names)
    p["mu0"] = jnp.zeros((Z,)); p["logP0"] = jnp.full((Z,), 1.0)
    p["W"] = -0.05 * jnp.eye(Z) + 0.01 * jax.random.normal(ks[o], (Z, Z))
    p["Wx"] = jnp.full((Z, Z), -4.0)
    p["B"] = jnp.full((Z, nE), -3.0) + 0.1 * jax.random.normal(ks[o + 1], (Z, nE))
    p["c"] = jnp.zeros((Z,)); p["q"] = jnp.full((Z,), -3.0)
    p["haz1"] = dense_init(ks[o + 2], Z, cfg["hid"]); p["haz2"] = dense_init(ks[o + 3], cfg["hid"], 1)
    if cfg["struct"] and cfg.get("mono", True):   # monotone read-out: raw weights pass through softplus, so start them small
        p["haz1"]["w"] = -3.0 + 0.5 * jax.random.normal(ks[o + 5], (Z, cfg["hid"]))
        p["haz1"]["b"] = jnp.full((cfg["hid"],), -1.0)
        p["haz2"]["w"] = -1.0 + 0.3 * jax.random.normal(ks[o + 6], (cfg["hid"], 1))
    p["alpha"] = jnp.full((K,), -4.0)
    p["beta"] = jnp.full((nE,), -4.0)
    p["dec"] = mlp_init(ks[o + 4], [Z, 64, C["tgt"].shape[-1]])
    if not cfg["pretrained"]:
        p["enc"] = enc_params
    return p


def _W(p, mk, struct):
    if struct:
        return p["W"] * mk["Wd"] + jax.nn.softplus(p["Wx"]) * mk["Wx"] * 0.5
    return p["W"] * mk["Wd"]


def _B(p, mk, struct):
    if struct:
        return jax.nn.softplus(p["B"]) * mk["B"] * mk["sign"][None, :]
    return p["B"] * 0.2


def drift(p, mk, cfg, s, u):
    return s @ _W(p, mk, cfg["struct"]).T + u @ _B(p, mk, cfg["struct"]).T + p["c"]


def hazard_logit(p, mk, cfg, s, u):
    if cfg["struct"] and cfg.get("mono", True):
        h1 = jax.nn.softplus(s @ jax.nn.softplus(p["haz1"]["w"]) + p["haz1"]["b"])
        out = (h1 @ jax.nn.softplus(p["haz2"]["w"]))[..., 0]
        direct = (u * (jax.nn.softplus(p["beta"]) * mk["sign"])).sum(-1)
    else:
        h1 = jax.nn.gelu(dense(p["haz1"], s))
        out = dense(p["haz2"], h1)[..., 0]
        if cfg["struct"]:
            direct = (u * (jax.nn.softplus(p["beta"]) * mk["sign"])).sum(-1)
        else:
            direct = (u * p["beta"] * 0.2).sum(-1)
    return out + direct


def qvec(p, cfg):
    return jax.nn.softplus(p["q"]) if cfg["stale"] else jnp.zeros_like(p["q"])


def filter_(p, mk, cfg, emb, mm, t, valid, ue, um, L):
    """emb (B,R,M,D), mm (B,R,M), t/valid (B,R), ue (B,R,nE) standardized exposures, um (B,R) exposure observed.
    Returns posterior at landmark and the one-step priors at each record."""
    Bsz, R, M, _ = emb.shape
    Z = cfg["Z"]
    names_n = M
    ob = []
    names = list(p["adapt"])
    for j, m in enumerate(names):
        pa = p["adapt"][m]
        y = mlp(pa, emb[:, :, j]) if isinstance(pa, list) else dense(pa, emb[:, :, j])
        ob.append(y)
    ob = jnp.stack(ob, 2)                                            # B,R,M,2Z
    o, lp = ob[..., :Z], ob[..., Z:]
    pi = jax.nn.softplus(lp) * mk["obs"][None, None] * mm[..., None]     # B,R,M,Z
    q = qvec(p, cfg)
    nE = ue.shape[-1]

    def pred(mu, P, u, dt):
        h = dt / cfg["sub"]
        for _ in range(cfg["sub"]):
            mu = jnp.clip(mu + h[:, None] * drift(p, mk, cfg, mu, u), -12, 12)
            P = P + h[:, None] * q
        return mu, P

    def step(carry, x):
        mu, P, u, tl = carry
        tk, vk, ok, pik, uk, umk = x
        dt = jnp.where(vk, jnp.maximum(tk - tl, 0.0), 0.0)
        mu_p, P_p = pred(mu, P, u, dt)
        lam = 1.0 / P_p + pik.sum(1)
        mu_n = (mu_p / P_p + (pik * ok).sum(1)) / lam
        P_n = 1.0 / lam
        v = vk[:, None]
        mu_n = jnp.where(v, mu_n, mu); P_n = jnp.where(v, P_n, P)
        u_n = jnp.where((umk & vk)[:, None], uk, u)
        tl_n = jnp.where(vk, tk, tl)
        return (mu_n, P_n, u_n, tl_n), (mu_p, P_p)

    mu0 = jnp.broadcast_to(p["mu0"], (Bsz, Z)); P0 = jnp.broadcast_to(jnp.exp(p["logP0"]), (Bsz, Z))
    carry = (mu0, P0, jnp.zeros((Bsz, nE)), jnp.zeros((Bsz,)))
    xs = (t.T, valid.T, jnp.moveaxis(o, 1, 0), jnp.moveaxis(pi, 1, 0), jnp.moveaxis(ue, 1, 0), um.T)
    (mu, P, u, tl), (mup, Pp) = jax.lax.scan(step, carry, xs)
    mu, P = pred(mu, P, u, jnp.maximum(L - tl, 0.0))
    return mu, P, u, jnp.moveaxis(mup, 0, 1)


def forecast(p, mk, cfg, mu, P, u, key, dt, K, S):
    """Monte Carlo roll-out -> per-sample discrete hazards (S,B,K)."""
    Bsz, Z = mu.shape
    k0, k1 = jax.random.split(key)
    e0 = jax.random.normal(k0, (S, Bsz, Z)) if cfg["sample"] else jnp.zeros((S, Bsz, Z))
    eps = jax.random.normal(k1, (K, S, Bsz, Z)) if cfg["sample"] else jnp.zeros((K, S, Bsz, Z))
    s0 = mu[None] + jnp.sqrt(P)[None] * e0
    q = qvec(p, cfg)

    def step(s, xs):
        a, e = xs
        h = jax.nn.sigmoid(a + hazard_logit(p, mk, cfg, s, u[None]))
        if cfg.get("roll", True):
            s = jnp.clip(s + dt * drift(p, mk, cfg, s, u[None]) + jnp.sqrt(q * dt) * e, -12, 12)
        return s, h

    _, hs = jax.lax.scan(step, s0, (p["alpha"], eps))
    return jnp.moveaxis(hs, 0, 2)                                    # S,B,K


def surv_loglik(h, jidx, ev):
    """Discrete-time mixture log-likelihood. h (S,B,K); jidx (B,) interval of event/censoring (K = beyond grid)."""
    S, B, K = h.shape
    lS = jnp.cumsum(jnp.log1p(-jnp.clip(h, 1e-6, 1 - 1e-6)), -1)       # log P(T > end of interval k)
    lS_prev = jnp.concatenate([jnp.zeros((S, B, 1)), lS], -1)         # log P(T > start of interval k)
    jj = jnp.clip(jidx, 0, K)
    lsp = jnp.take_along_axis(lS_prev, jj[None, :, None].repeat(S, 0), 2)[..., 0]
    hj = jnp.take_along_axis(jnp.concatenate([h, jnp.ones((S, B, 1)) * 0.5], -1), jj[None, :, None].repeat(S, 0), 2)[..., 0]
    ll = jnp.where(ev[None] > 0, lsp + jnp.log(jnp.clip(hj, 1e-6, 1.0)), lsp)
    return jax.scipy.special.logsumexp(ll, 0) - jnp.log(S)


def cif(h):
    lS = jnp.cumsum(jnp.log1p(-jnp.clip(h, 1e-6, 1 - 1e-6)), -1)
    return 1.0 - jnp.exp(lS).mean(0)                                  # B,K


# ------------------------------------------------------------------------------------------ data feeding
class Feeder:
    def __init__(self, C, emb, cfg, emb_stats=None):
        self.C, self.cfg = C, cfg
        names = list(C["mods"])
        self.mm = np.stack([C["mods"][m]["mask"] for m in names], -1).astype(np.float32)
        if emb is not None and np.ndim(emb) == 4:
            # per-modality, per-dimension standardisation of frozen embeddings (training records only)
            if emb_stats is None:
                tr = C["split"] == 0
                mu = np.zeros(emb.shape[2:], np.float32); sd = np.ones(emb.shape[2:], np.float32)
                for j in range(emb.shape[2]):
                    sel = emb[tr][self.mm[tr][..., j] > 0][:, j]
                    mu[j], sd[j] = sel.mean(0), sel.std(0) + 1e-6
                emb_stats = (mu, sd)
            self.emb_stats = emb_stats
            emb = ((emb - emb_stats[0]) / emb_stats[1] * self.mm[..., None]).astype(np.float32)
        self.emb = emb
        idx = C["expo_idx"]
        ue = (C["u"][..., idx] - C["u_mu"][idx]) / C["u_sd"][idx]
        um = C["u_mask"].copy()
        # steps from wearable windows update the activity exposure between visits
        if "wear_steps" in C:
            js = C["expo_names"].index("steps")
            ws = (C["wear_steps"] - C["u_mu"][idx[js]]) / C["u_sd"][idx[js]]
            only_w = C["wear_mask"] & ~um
            ue[..., js] = np.where(C["wear_mask"], ws, ue[..., js])
            # wear-only records: keep previously observed values for other exposures by marking them partial
            self.wear_only = only_w
        else:
            self.wear_only = np.zeros_like(um)
        self.ue, self.um = ue.astype(np.float32), um
        self.ipw = np.ones(len(C["T"]), np.float32)

    def batch(self, rows, L, raw=False):
        C, cfg = self.C, self.cfg
        L = np.broadcast_to(np.asarray(L, np.float32), (len(rows),))
        t = C["t"][rows]
        valid = C["valid"][rows] & (t <= L[:, None] + 1e-6)
        mm = self.mm[rows] * valid[..., None]
        if not cfg["longitudinal"]:
            # keep only the most recent clinic visit and its modalities
            vis = valid & C["visit"][rows]
            last = np.where(vis.any(1), (np.arange(t.shape[1])[None] * vis).max(1), -1)
            keep = np.arange(t.shape[1])[None] == last[:, None]
            valid = valid & keep; mm = mm * keep[..., None]
        ue = self.ue[rows].copy()
        um = self.um[rows] & valid
        # wear-only records carry the previous exposure values except steps
        wo = self.wear_only[rows] & valid
        if wo.any() and "steps" in C["expo_names"]:
            js = C["expo_names"].index("steps")
            for r in range(t.shape[1]):
                prev = ue[:, r - 1] if r > 0 else np.zeros_like(ue[:, 0])
                rowsw = wo[:, r]
                tmp = prev[rowsw].copy(); tmp[:, js] = ue[rowsw, r, js]
                ue[rowsw, r] = tmp
            um = um | wo
        Trel = C["T"][rows] - L
        jidx = np.minimum(np.floor(Trel / C["grid_dt"]).astype(np.int32), C["K"])
        ev = ((C["E"][rows] > 0) & (jidx < C["K"])).astype(np.float32)
        out = dict(mm=mm.astype(np.float32), t=t.astype(np.float32), valid=valid, ue=ue, um=um, L=L,
                   jidx=jidx, ev=ev, w=self.ipw[rows], tgt=C["tgt"][rows], tmask=C["tmask"][rows] * valid[..., None])
        if raw:
            out["raw"] = {m: C["mods"][m]["X"][rows] for m in C["mods"]}
        else:
            out["emb"] = self.emb[rows]
        return out


def at_risk(C, rows, L):
    has = (C["valid"][rows] & (C["t"][rows] <= L + 1e-6)).any(1)
    return np.asarray(rows)[(C["T"][rows] > L) & has]


# ------------------------------------------------------------------------------------------ model API
class CardioTwin:
    def __init__(self, C, cfg=None, seed=0, enc_params=None, enc_kinds=None):
        self.C = C
        self.cfg = dict(DEFAULT, **(cfg or {}))
        self.mk = masks(self.cfg, C)
        self.key = jax.random.PRNGKey(seed)
        self.seed = seed
        self.kinds = {m: C["mods"][m]["kind"] for m in C["mods"]}
        self.p = init(jax.random.PRNGKey(seed + 17), self.cfg, C, enc_params=enc_params)
        self._build()

    def _embed(self, p, b):
        if self.cfg["pretrained"]:
            return b["emb"]
        outs = []
        for m in self.C["mods"]:
            x = b["raw"][m]
            Bsz, R = x.shape[:2]
            e = enc_apply(p["enc"][m], self.kinds[m], x.reshape((Bsz * R,) + x.shape[2:]))
            outs.append(e.reshape(Bsz, R, -1))
        return jnp.stack(outs, 2)

    def _build(self):
        cfg, mk, C = self.cfg, self.mk, self.C
        dt, K = C["grid_dt"], C["K"]

        def loss(p, b, key):
            emb = self._embed(p, b)
            mu, P, u, mup = filter_(p, mk, cfg, emb, b["mm"], b["t"], b["valid"], b["ue"], b["um"], b["L"])
            h = forecast(p, mk, cfg, mu, P, u, key, dt, K, cfg["S"])
            ll = surv_loglik(h, b["jidx"], b["ev"])
            w = b["w"] if cfg["ipw"] else jnp.ones_like(b["w"])
            nll = -(w * ll).sum() / w.sum()
            pred = mlp(p["dec"], mup)
            tm = b["tmask"].at[:, 0].set(0.0)
            dec = (tm * (pred - b["tgt"]) ** 2).sum() / jnp.maximum(tm.sum(), 1.0)
            reg = 1e-4 * (jnp.sum(p["W"] ** 2))
            pen = 0.0
            if cfg["struct"] and not cfg["mono"] and cfg["nu"] > 0:
                # soft monotonicity: moving every exposure in its harmful direction must not lower cumulative hazard
                h2 = forecast(p, mk, cfg, mu, P, u + cfg["eps_mono"] * mk["sign"][None, :], key, dt, K, cfg["S"])
                H1 = -jnp.log1p(-jnp.clip(h, 1e-6, 1 - 1e-6)).cumsum(-1)
                H2 = -jnp.log1p(-jnp.clip(h2, 1e-6, 1 - 1e-6)).cumsum(-1)
                pen = jax.nn.relu(H1 - H2).mean() / cfg["eps_mono"]
            return nll + cfg["lam_dec"] * dec + reg + cfg["nu"] * pen, (nll, dec)

        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(cfg["lr"], weight_decay=cfg["wd"]))
        self.opt = opt

        @jax.jit
        def step(p, st, b, key):
            (l, aux), g = jax.value_and_grad(loss, has_aux=True)(p, b, key)
            up, st = opt.update(g, st, p)
            return optax.apply_updates(p, up), st, l, aux

        @jax.jit
        def evalnll(p, b, key):
            return loss(p, b, key)[1][0]

        @jax.jit
        def predict(p, b, key, delta, setm, setv):
            emb = self._embed(p, b)
            mu, P, u, _ = filter_(p, mk, cfg, emb, b["mm"], b["t"], b["valid"], b["ue"], b["um"], b["L"])
            u2 = jnp.where(setm > 0, setv, u + delta)
            h = forecast(p, mk, cfg, mu, P, u2, key, dt, K, 32)
            return cif(h), mu, P

        self._step, self._evalnll, self._predict = step, evalnll, predict

    def _jb(self, b):
        return {k: (jax.tree_util.tree_map(jnp.asarray, v) if isinstance(v, dict) else jnp.asarray(v)) for k, v in b.items()}

    def fit(self, feeder, train_rows, val_rows, verbose=False):
        cfg, C = self.cfg, self.C
        st = self.opt.init(self.p)
        rng = np.random.default_rng(self.seed)
        Ls = C["landmarks"]
        # landmark super-model: each (patient, landmark) pair at risk is a training example
        pairs = [(r, L) for L in Ls for r in at_risk(C, train_rows, L)]
        pr = np.array([a for a, _ in pairs]); pl = np.array([b for _, b in pairs], np.float32)
        vrows = at_risk(C, val_rows, C["main_landmark"])
        best, best_p, bad = np.inf, self.p, 0
        raw = not cfg["pretrained"]
        hist = []
        for ep in range(cfg["epochs"]):
            perm = rng.permutation(len(pr))
            for bix in range(0, len(perm) - cfg["batch"] + 1, cfg["batch"]):
                ix = perm[bix:bix + cfg["batch"]]
                b = self._jb(feeder.batch(pr[ix], pl[ix], raw=raw))
                self.key, k = jax.random.split(self.key)
                self.p, st, l, aux = self._step(self.p, st, b, k)
            v = []
            for bix in range(0, len(vrows), 512):
                b = self._jb(feeder.batch(vrows[bix:bix + 512], C["main_landmark"], raw=raw))
                v.append(float(self._evalnll(self.p, b, jax.random.PRNGKey(1))) * len(vrows[bix:bix + 512]))
            vl = sum(v) / len(vrows)
            hist.append(vl)
            if verbose:
                print(f"    ep {ep} train {float(l):.4f} val {vl:.4f}", flush=True)
            if vl < best - 1e-4:
                best, best_p, bad = vl, self.p, 0
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        self.p = best_p
        self.hist = hist
        return self

    def predict(self, feeder, rows, L, delta=None, setm=None, setv=None, seed=123):
        """Returns CIF on the full grid (n,K). delta: (nE,) shift of standardized exposures applied from the
        landmark onward; setm/setv: (nE,) mask and standardized value for exposures fixed by do(.)."""
        C = self.C
        nE = len(C["expo_idx"])
        sm = jnp.zeros((nE,)) if setm is None else jnp.asarray(setm, jnp.float32)
        sv = jnp.zeros((nE,)) if setv is None else jnp.asarray(setv, jnp.float32)
        out = []
        raw = not self.cfg["pretrained"]
        for bix in range(0, len(rows), 512):
            rr = rows[bix:bix + 512]
            b = self._jb(feeder.batch(rr, L, raw=raw))
            d = jnp.zeros((nE,)) if delta is None else jnp.asarray(delta, jnp.float32)
            F, mu, P = self._predict(self.p, b, jax.random.PRNGKey(seed), d, sm, sv)
            out.append(np.asarray(F))
        return np.concatenate(out, 0)

    def state(self, feeder, rows, L):
        mus, Ps = [], []
        raw = not self.cfg["pretrained"]
        for bix in range(0, len(rows), 512):
            b = self._jb(feeder.batch(rows[bix:bix + 512], L, raw=raw))
            z = jnp.zeros((len(self.C["expo_idx"]),))
            _, mu, P = self._predict(self.p, b, jax.random.PRNGKey(0), z, z, z)
            mus.append(np.asarray(mu)); Ps.append(np.asarray(P))
        return np.concatenate(mus), np.concatenate(Ps)
