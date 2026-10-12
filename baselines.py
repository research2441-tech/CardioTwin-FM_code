"""Comparison models: Cox PH, random survival forest, DeepSurv, DeepHit (landmark features),
Dynamic-DeepHit (GRU over records) and a multimodal Transformer over pretrained modality tokens."""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from nets import mlp_init, mlp, dense_init, dense, layernorm, enc_apply
from features import landmark_features, record_features
from twin import at_risk


# ----------------------------------------------------------------------------------------- data
def stacked_landmark_rows(C, rows):
    pairs = [(r, L) for L in C["landmarks"] for r in at_risk(C, rows, L)]
    return np.array([a for a, _ in pairs]), np.array([b for _, b in pairs], np.float32)


class StaticData:
    def __init__(self, C):
        self.C = C
        self.RF = record_features(C)

    def X(self, rows, L):
        L = np.broadcast_to(np.asarray(L, np.float32), (len(rows),))
        out = []
        for Lv in np.unique(L):
            m = L == Lv
            out.append((np.where(m)[0], landmark_features(self.C, rows[m], float(Lv), self.RF)))
        X = np.zeros((len(rows), out[0][1].shape[1] + 1), np.float32)
        for ix, f in out:
            X[ix, :-1] = f
        X[:, -1] = L
        return X


def to_grid_cif(surv_fns, grid):
    return np.stack([1 - fn(np.minimum(grid, fn.x[-1])) for fn in surv_fns], 0)


# ----------------------------------------------------------------------------------------- sksurv models
class SkModel:
    def __init__(self, kind, seed=0):
        self.kind, self.seed = kind, seed

    def fit(self, Xtr, ttr, etr):
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Z = (Xtr - self.mu) / self.sd
        y = Surv.from_arrays(etr > 0, ttr)
        if self.kind == "cox":
            self.m = CoxPHSurvivalAnalysis(alpha=1.0, ties="efron").fit(Z, y)
        else:
            self.m = RandomSurvivalForest(n_estimators=100, min_samples_leaf=40, max_features="sqrt", n_jobs=2,
                                          random_state=self.seed, low_memory=False).fit(Z, y)
        return self

    def cif(self, X, grid):
        Z = (X - self.mu) / self.sd
        return to_grid_cif(self.m.predict_survival_function(Z), grid)


# ----------------------------------------------------------------------------------------- JAX losses
def deephit_loss(logits, jidx, ev, K, alpha=0.2, sigma=0.1):
    """logits (B,K+1) -> PMF over K grid intervals plus 'beyond'."""
    pmf = jax.nn.softmax(logits, -1)
    F = jnp.cumsum(pmf, -1)
    B = logits.shape[0]
    j = jnp.clip(jidx, 0, K)
    pj = pmf[jnp.arange(B), j]
    surv_from = 1.0 - jnp.concatenate([jnp.zeros((B, 1)), F[:, :-1]], -1)[jnp.arange(B), j]
    nll = -jnp.where(ev > 0, jnp.log(pj + 1e-8), jnp.log(surv_from + 1e-8)).mean()
    Fi = F[jnp.arange(B), jnp.clip(j, 0, K - 1)]                       # F_i(T_i)
    Fk = F[:, jnp.clip(j, 0, K - 1)].T                                 # F_k(T_i) -> (B_i, B_k)
    acc = (ev[:, None] > 0) & (jidx[:, None] < jidx[None, :])
    rank = (acc * jnp.exp(-(Fi[:, None] - Fk) / sigma)).sum() / jnp.maximum(acc.sum(), 1)
    return nll + alpha * rank


def cox_loss(score, t, e):
    o = jnp.argsort(-t)
    s, ee = score[o], e[o]
    lse = jax.lax.cumlogsumexp(s)
    return -((s - lse) * ee).sum() / jnp.maximum(ee.sum(), 1)


def train_jax(params, loss_fn, batches_fn, val_fn, epochs=40, lr=1e-3, patience=6, seed=0):
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
    st = opt.init(params)

    @jax.jit
    def step(p, st, b, k):
        l, g = jax.value_and_grad(loss_fn)(p, b, k)
        up, st = opt.update(g, st, p)
        return optax.apply_updates(p, up), st, l

    key = jax.random.PRNGKey(seed)
    best, bp, bad = np.inf, params, 0
    for ep in range(epochs):
        for b in batches_fn(ep):
            key, k = jax.random.split(key)
            params, st, l = step(params, st, b, k)
        v = val_fn(params)
        if v < best - 1e-4:
            best, bp, bad = v, params, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return bp


def grid_targets(C, rows, L):
    Trel = C["T"][rows] - L
    jidx = np.minimum(np.floor(Trel / C["grid_dt"]).astype(np.int32), C["K"])
    ev = ((C["E"][rows] > 0) & (jidx < C["K"])).astype(np.float32)
    return jidx, ev, Trel


# ----------------------------------------------------------------------------------------- DeepSurv / DeepHit
class DeepStatic:
    def __init__(self, C, kind, seed=0, hidden=(128, 64)):
        self.C, self.kind, self.seed = C, kind, seed
        self.hidden = hidden

    def fit(self, Xtr, jtr, etr, ttr, Xva, jva, eva, tva):
        C = self.C
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Z, Zv = (Xtr - self.mu) / self.sd, (Xva - self.mu) / self.sd
        dout = 1 if self.kind == "deepsurv" else C["K"] + 1
        p = mlp_init(jax.random.PRNGKey(self.seed), [Z.shape[1], *self.hidden, dout])
        K = C["K"]
        rng = np.random.default_rng(self.seed)

        def fwd(p, x, key=None, train=False):
            h = x
            for i, l in enumerate(p):
                h = dense(l, h)
                if i < len(p) - 1:
                    h = jax.nn.gelu(h)
                    if train:
                        key, k = jax.random.split(key)
                        h = h * jax.random.bernoulli(k, 0.8, h.shape) / 0.8
            return h

        if self.kind == "deepsurv":
            loss = lambda p, b, k: cox_loss(fwd(p, b[0], k, True)[:, 0], b[1], b[2])
            vb = (jnp.asarray(Zv), jnp.asarray(tva), jnp.asarray(eva))
            val = lambda p: float(cox_loss(fwd(p, vb[0])[:, 0], vb[1], vb[2]))
            data = (Z, ttr, etr.astype(np.float32))
        else:
            loss = lambda p, b, k: deephit_loss(fwd(p, b[0], k, True), b[1], b[2], K)
            vb = (jnp.asarray(Zv), jnp.asarray(jva), jnp.asarray(eva))
            val = lambda p: float(deephit_loss(fwd(p, vb[0]), vb[1], vb[2], K, alpha=0.0))
            data = (Z, jtr, etr)

        def batches(ep):
            perm = rng.permutation(len(Z))
            for b in range(0, len(perm) - 255, 256):
                ix = perm[b:b + 256]
                yield tuple(jnp.asarray(d[ix]) for d in data)

        self.p = train_jax(p, loss, batches, val, seed=self.seed)
        self.fwd = jax.jit(lambda p, x: fwd(p, x))
        if self.kind == "deepsurv":       # Breslow baseline cumulative hazard
            s = np.asarray(self.fwd(self.p, jnp.asarray(Z)))[:, 0]
            o = np.argsort(ttr)
            ts, es, ss = ttr[o], etr[o], np.exp(s[o])
            risk_sum = np.cumsum(ss[::-1])[::-1]
            self.bt = ts[es > 0]
            self.bh = np.cumsum(1.0 / risk_sum[es > 0])
        return self

    def cif(self, X, grid):
        Z = (X - self.mu) / self.sd
        out = np.asarray(self.fwd(self.p, jnp.asarray(Z)))
        if self.kind == "deepsurv":
            H0 = np.interp(grid, self.bt, self.bh, left=0.0)
            return 1 - np.exp(-H0[None] * np.exp(out[:, :1]))
        pmf = np.exp(out - out.max(1, keepdims=True)); pmf /= pmf.sum(1, keepdims=True)
        return np.cumsum(pmf, 1)[:, :self.C["K"]]


# ----------------------------------------------------------------------------------------- Dynamic-DeepHit
def gru_init(key, din, dh):
    k = jax.random.split(key, 2)
    return {"x": dense_init(k[0], din, 3 * dh), "h": dense_init(k[1], dh, 3 * dh)}


def gru_cell(p, h, x):
    gx, gh = dense(p["x"], x), dense(p["h"], h)
    d = h.shape[-1]
    r = jax.nn.sigmoid(gx[..., :d] + gh[..., :d])
    z = jax.nn.sigmoid(gx[..., d:2 * d] + gh[..., d:2 * d])
    n = jnp.tanh(gx[..., 2 * d:] + r * gh[..., 2 * d:])
    return (1 - z) * n + z * h


class SeqModel:
    """kind='ddh': GRU over engineered record features; kind='mmt': Transformer over pretrained tokens."""
    def __init__(self, C, kind, seed=0, emb=None, d=64):
        self.C, self.kind, self.seed, self.emb, self.d = C, kind, seed, emb, d
        self.RF = record_features(C) if kind == "ddh" else None
        if self.RF is not None:
            tr = C["split"] == 0
            v = C["valid"][tr]
            self.rmu = self.RF[tr][v].mean(0); self.rsd = self.RF[tr][v].std(0) + 1e-6
        self.mm = np.stack([C["mods"][m]["mask"] for m in C["mods"]], -1).astype(np.float32)
        idx = C["expo_idx"] + C.get("treat_idx", [])
        self.ue = ((C["u"][..., idx] - C["u_mu"][idx]) / C["u_sd"][idx]).astype(np.float32)

    def inputs(self, rows, L, emb=None):
        C = self.C
        L = np.broadcast_to(np.asarray(L, np.float32), (len(rows),))
        t = C["t"][rows]
        valid = C["valid"][rows] & (t <= L[:, None] + 1e-6)
        dtL = ((L[:, None] - t) * valid).astype(np.float32)
        if self.kind == "ddh":
            x = (self.RF[rows] - self.rmu) / self.rsd * valid[..., None]
            x = np.concatenate([x, dtL[..., None], valid[..., None].astype(np.float32)], -1)
            return dict(x=x.astype(np.float32), valid=valid)
        E = self.emb if emb is None else emb
        return dict(e=E[rows], mm=self.mm[rows] * valid[..., None], u=self.ue[rows] * (C["u_mask"][rows] & valid)[..., None],
                    um=(C["u_mask"][rows] & valid).astype(np.float32), dtL=dtL, valid=valid)

    def _init(self, din=None):
        key = jax.random.PRNGKey(self.seed)
        ks = jax.random.split(key, 12)
        K, d = self.C["K"], self.d
        if self.kind == "ddh":
            return {"gru": gru_init(ks[0], din, d), "att": dense_init(ks[1], d, 1),
                    "head": mlp_init(ks[2], [2 * d, 64, K + 1])}
        M = self.mm.shape[-1]
        nu = self.ue.shape[-1]
        p = {"proj": dense_init(ks[0], self.emb.shape[-1], d), "uproj": dense_init(ks[1], nu, d),
             "mtok": 0.02 * jax.random.normal(ks[2], (M + 1, d)), "tproj": dense_init(ks[3], 8, d),
             "head": mlp_init(ks[4], [d, 64, K + 1]), "layers": []}
        for i in range(2):
            kk = jax.random.split(ks[5 + i], 4)
            p["layers"].append({"qkv": dense_init(kk[0], d, 3 * d), "o": dense_init(kk[1], d, d),
                                "ff": mlp_init(kk[2], [d, 2 * d, d])})
        return p

    def _fwd(self, p, b, key=None, train=False):
        if self.kind == "ddh":
            x, valid = b["x"], b["valid"]

            def step(h, xs):
                xt, vt = xs
                hn = gru_cell(p["gru"], h, xt)
                return jnp.where(vt[:, None], hn, h), jnp.where(vt[:, None], hn, 0.0)
            h0 = jnp.zeros((x.shape[0], self.d))
            hl, hs = jax.lax.scan(step, h0, (jnp.moveaxis(x, 1, 0), valid.T))
            hs = jnp.moveaxis(hs, 0, 1)
            a = dense(p["att"], hs)[..., 0]
            a = jnp.where(valid, a, -1e4)
            w = jax.nn.softmax(a, 1)
            ctx = (w[..., None] * hs).sum(1)
            return mlp(p["head"], jnp.concatenate([hl, ctx], -1))
        e, mm, u, um, dtL = b["e"], b["mm"], b["u"], b["um"], b["dtL"]
        Bsz, R, M, _ = e.shape
        if train:
            drop = jax.random.bernoulli(key, 0.8, mm.shape)
            mm = mm * drop
        freqs = jnp.array([0.25, 0.5, 1, 2, 4, 8, 16, 32.0])
        te = dense(p["tproj"], jnp.sin(dtL[..., None] / freqs))                       # B,R,d
        tok = dense(p["proj"], e) + p["mtok"][None, None, :M] + te[:, :, None]
        utok = dense(p["uproj"], u) + p["mtok"][M] + te
        X = jnp.concatenate([tok.reshape(Bsz, R * M, -1), utok], 1)
        msk = jnp.concatenate([mm.reshape(Bsz, R * M), um], 1)
        d = X.shape[-1]; H = 4
        for lyr in p["layers"]:
            hN = layernorm(X)
            qkv = dense(lyr["qkv"], hN).reshape(Bsz, -1, 3, H, d // H)
            q, k_, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            att = jnp.einsum("bihd,bjhd->bhij", q, k_) / np.sqrt(d // H)
            att = jnp.where(msk[:, None, None, :] > 0, att, -1e4)
            att = jax.nn.softmax(att, -1)
            X = X + dense(lyr["o"], jnp.einsum("bhij,bjhd->bihd", att, v).reshape(Bsz, -1, d))
            X = X + mlp(lyr["ff"], layernorm(X))
        pooled = (X * msk[..., None]).sum(1) / jnp.maximum(msk.sum(1, keepdims=True), 1)
        return mlp(p["head"], layernorm(pooled))

    def fit(self, train_rows, val_rows):
        C, K = self.C, self.C["K"]
        pr, pl = stacked_landmark_rows(C, train_rows)
        vr = at_risk(C, val_rows, C["main_landmark"])
        jtr, etr, _ = grid_targets(C, pr, pl)
        jva, eva, _ = grid_targets(C, vr, C["main_landmark"])
        b0 = self.inputs(pr[:2], pl[:2])
        p = self._init(b0["x"].shape[-1] if self.kind == "ddh" else None)
        rng = np.random.default_rng(self.seed)
        loss = lambda p, b, k: deephit_loss(self._fwd(p, b[0], k, True), b[1], b[2], K)
        vin = {k: jnp.asarray(v) for k, v in self.inputs(vr, C["main_landmark"]).items()}
        vlossf = jax.jit(lambda p: deephit_loss(self._fwd(p, vin), jnp.asarray(jva), jnp.asarray(eva), K, alpha=0.0))
        val = lambda p: float(vlossf(p))

        def batches(ep):
            perm = rng.permutation(len(pr))
            for b in range(0, len(perm) - 127, 128):
                ix = perm[b:b + 128]
                inp = {k: jnp.asarray(v) for k, v in self.inputs(pr[ix], pl[ix]).items()}
                yield (inp, jnp.asarray(jtr[ix]), jnp.asarray(etr[ix]))

        self.p = train_jax(p, loss, batches, val, epochs=30, lr=1e-3, seed=self.seed)
        self._jf = jax.jit(lambda p, b: self._fwd(p, b))
        return self

    def cif(self, rows, L, emb=None, ue_override=None):
        out = []
        for b in range(0, len(rows), 512):
            rr = rows[b:b + 512]
            inp = self.inputs(rr, L, emb)
            if ue_override is not None:
                inp["u"] = ue_override(rr, inp)
            o = np.asarray(self._jf(self.p, {k: jnp.asarray(v) for k, v in inp.items()}))
            pmf = np.exp(o - o.max(1, keepdims=True)); pmf /= pmf.sum(1, keepdims=True)
            out.append(np.cumsum(pmf, 1)[:, :self.C["K"]])
        return np.concatenate(out, 0)
