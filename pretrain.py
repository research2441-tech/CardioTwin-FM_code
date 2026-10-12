"""Stage 1: self-supervised multimodal pretraining with modality-aware masking.

For every clinical record with at least two available modalities, one available
modality is hidden; the precision-free mean of the remaining projected tokens
(context) must (i) identify the hidden modality's embedding among the batch
(InfoNCE, cross-modal alignment) and (ii) reconstruct a low-dimensional summary
of the hidden raw signal. Records with a single modality contribute only the
reconstruction of a randomly dropped half of their own features.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from nets import enc_init, enc_apply, mlp_init, mlp, dense_init, dense

DEMB = 64


def summary_target(kind, X):
    """Low-dimensional reconstruction target of a raw modality (numpy or jax arrays)."""
    if kind == "wave":                   # (.., C, L) -> 10 window means per channel
        L = X.shape[-1]
        w = X.reshape(X.shape[:-1] + (10, L // 10)).mean(-1)
        return w.reshape(X.shape[:-2] + (-1,))
    if kind == "seq":
        return jnp.concatenate([X.mean(-1), jnp.sqrt(X.var(-1) + 1e-6)], -1)
    return X


def init(key, C):
    params = {"enc": {}, "proj": {}, "dec": {}, "mtok": {}}
    names = list(C["mods"])
    keys = jax.random.split(key, 4 * len(names) + 2)
    for i, m in enumerate(names):
        sp = C["mods"][m]
        X = sp["X"]
        din = X.shape[2] if sp["kind"] in ("wave", "seq") else X.shape[-1]
        params["enc"][m] = enc_init(keys[4 * i], sp["kind"], din, DEMB)
        params["proj"][m] = dense_init(keys[4 * i + 1], DEMB, DEMB)
        tdim = np.asarray(summary_target(sp["kind"], X[:1, :1])).shape[-1]
        params["dec"][m] = mlp_init(keys[4 * i + 2], [DEMB, 128, tdim])
        params["mtok"][m] = 0.02 * jax.random.normal(keys[4 * i + 3], (DEMB,))
        params.setdefault("own", {})[m] = mlp_init(jax.random.fold_in(keys[4 * i + 2], 7), [DEMB, 128, tdim])
    params["pred"] = mlp_init(keys[-1], [DEMB, 128, DEMB])
    return params


def encode_all(params, C, X_by_mod):
    return {m: enc_apply(params["enc"][m], C["mods"][m]["kind"], X_by_mod[m]) for m in X_by_mod}


def make_loss(C, tau=0.1):
    names = list(C["mods"])
    kinds = {m: C["mods"][m]["kind"] for m in names}

    def loss(params, Xb, Mb, key):
        # Xb[m]: (B, ...) raw ; Mb: (B, M) availability
        B = Mb.shape[0]
        kmask, key = jax.random.split(key)
        # intra-modal masking (MAE-style): hide a random contiguous 20% segment of every waveform/series,
        # and a random 20% of tabular entries, before encoding
        Xin = {}
        for i, m in enumerate(names):
            x = Xb[m]
            km = jax.random.fold_in(kmask, i)
            if kinds[m] in ("wave", "seq"):
                Lx = x.shape[-1]; w = max(1, Lx // 5)
                st = jax.random.randint(km, (B, 1, 1), 0, Lx - w)
                pos = jnp.arange(Lx)[None, None, :]
                Xin[m] = jnp.where((pos >= st) & (pos < st + w), 0.0, x)
            else:
                Xin[m] = x * jax.random.bernoulli(km, 0.8, x.shape)
        emb = {m: enc_apply(params["enc"][m], kinds[m], Xin[m]) for m in names}
        Z = jnp.stack([dense(params["proj"][m], emb[m]) + params["mtok"][m] for m in names], 1)     # (B,M,D)
        # choose one available modality to hide (uniform over available)
        g = jax.random.gumbel(key, Mb.shape) + jnp.where(Mb > 0, 0.0, -1e4)
        hide = jax.nn.one_hot(jnp.argmax(g, 1), len(names))                                     # (B,M)
        ctx_mask = Mb * (1 - hide)
        nctx = ctx_mask.sum(1, keepdims=True)
        ctx = (Z * ctx_mask[..., None]).sum(1) / jnp.maximum(nctx, 1.0)
        has_ctx = (nctx[:, 0] > 0).astype(jnp.float32)
        pred = mlp(params["pred"], ctx)
        target = (Z * hide[..., None]).sum(1)
        pn = pred / jnp.sqrt((pred ** 2).sum(-1, keepdims=True) + 1e-6)
        tn = target / jnp.sqrt((target ** 2).sum(-1, keepdims=True) + 1e-6)
        logits = pn @ tn.T / tau
        lab = jnp.arange(B)
        valid = has_ctx[:, None] * has_ctx[None, :]
        logits = jnp.where(valid > 0, logits, -50.0)
        nce = -(jax.nn.log_softmax(logits, 1)[lab, lab] * has_ctx).sum() / jnp.maximum(has_ctx.sum(), 1)
        # reconstruction of the hidden modality summary from context (or from own masked token if alone)
        rec, own = 0.0, 0.0
        for j, m in enumerate(names):
            tgt_own = summary_target(kinds[m], Xb[m])
            out_own = mlp(params["own"][m], emb[m])
            wj = Mb[:, j]
            own = own + (wj * ((out_own - tgt_own) ** 2).mean(-1)).sum() / jnp.maximum(wj.sum(), 1.0)
        for j, m in enumerate(names):
            tgt = summary_target(kinds[m], Xb[m])
            src = jnp.where(has_ctx[:, None] > 0, ctx, Z[:, j] * 0.5)
            out = mlp(params["dec"][m], src)
            w = hide[:, j]
            rec = rec + (w * ((out - tgt) ** 2).mean(-1)).sum() / jnp.maximum(w.sum(), 1.0)
        return nce + 0.5 * rec + own, (nce, rec + own)

    return loss


def gather_records(C, rows):
    """Flatten (patient, record) pairs with at least one modality into a list."""
    names = list(C["mods"])
    M = np.stack([C["mods"][m]["mask"] for m in names], -1)        # N,R,M
    sel = np.argwhere((M[rows].sum(-1) > 0))
    pr = np.asarray(rows)[sel[:, 0]]
    return pr, sel[:, 1], M[pr, sel[:, 1]].astype(np.float32)


def pretrain(C, seed=0, epochs=12, batch=256, lr=2e-3, rows=None, verbose=True):
    names = list(C["mods"])
    if rows is None:
        rows = np.where(np.isin(C["split"], [0, 1]))[0]
    pr, rr, M = gather_records(C, rows)
    key = jax.random.PRNGKey(seed)
    params = init(key, C)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
    st = opt.init(params)
    lossf = make_loss(C)

    @jax.jit
    def step(params, st, Xb, Mb, k):
        (l, aux), g = jax.value_and_grad(lossf, has_aux=True)(params, Xb, Mb, k)
        up, st = opt.update(g, st, params)
        return optax.apply_updates(params, up), st, l, aux

    rng = np.random.default_rng(seed)
    n = len(pr)
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = []
        for b in range(0, n - batch + 1, batch):
            ix = perm[b:b + batch]
            Xb = {m: jnp.asarray(C["mods"][m]["X"][pr[ix], rr[ix]], jnp.float32) for m in names}
            key, k = jax.random.split(key)
            params, st, l, aux = step(params, st, Xb, jnp.asarray(M[ix]), k)
            tot.append([float(l), float(aux[0]), float(aux[1])])
        hist.append(np.mean(tot, 0))
        if verbose:
            print(f"  pretrain ep {ep} loss {hist[-1][0]:.3f} nce {hist[-1][1]:.3f} rec {hist[-1][2]:.3f}", flush=True)
    return params, np.array(hist)


def embed(params, C, batch=2048):
    """Encode every (patient, record, modality) with the frozen pretrained encoders -> (N,R,M,DEMB)."""
    names = list(C["mods"])
    N, R = C["t"].shape
    out = np.zeros((N, R, len(names), DEMB), np.float32)
    f = {m: jax.jit(lambda p, x, m=m: enc_apply(p, C["mods"][m]["kind"], x)) for m in names}
    for j, m in enumerate(names):
        idx = np.argwhere(C["mods"][m]["mask"])
        for b in range(0, len(idx), batch):
            ii = idx[b:b + batch]
            x = jnp.asarray(C["mods"][m]["X"][ii[:, 0], ii[:, 1]], jnp.float32)
            out[ii[:, 0], ii[:, 1], j] = np.asarray(f[m](params["enc"][m], x))
    return out


def augment(emb, C, width=None):
    """Append the raw standardized tabular measurements to the frozen embedding of tabular modalities
    (zeros for waveform / sequence modalities), giving adapters a residual path to exact values."""
    names = list(C["mods"])
    width = width or max(C["mods"][m]["X"].shape[-1] for m in names if C["mods"][m]["kind"] == "tab")
    N, R, M, D = emb.shape
    out = np.zeros((N, R, M, D + width), np.float32)
    out[..., :D] = emb
    for j, m in enumerate(names):
        if C["mods"][m]["kind"] == "tab":
            X = C["mods"][m]["X"]
            out[:, :, j, D:D + X.shape[-1]] = X
    return out
