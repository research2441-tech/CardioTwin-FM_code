"""Minimal JAX layers (no framework dependency beyond jax/optax)."""
import jax
import jax.numpy as jnp
import numpy as np


def glorot(key, shape, scale=1.0):
    fan_in, fan_out = shape[0], shape[-1]
    if len(shape) == 3:          # conv kernel (out, in, k)
        fan_in, fan_out = shape[1] * shape[2], shape[0] * shape[2]
    lim = scale * np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def dense_init(key, din, dout, scale=1.0):
    return {"w": glorot(key, (din, dout), scale), "b": jnp.zeros((dout,))}


def dense(p, x):
    return x @ p["w"] + p["b"]


def mlp_init(key, sizes, scale=1.0):
    keys = jax.random.split(key, len(sizes) - 1)
    return [dense_init(k, a, b, scale) for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def mlp(ps, x, act=jax.nn.gelu, final_act=False):
    for i, p in enumerate(ps):
        x = dense(p, x)
        if i < len(ps) - 1 or final_act:
            x = act(x)
    return x


def conv_init(key, cin, cout, k):
    return {"w": glorot(key, (cout, cin, k)), "b": jnp.zeros((cout,))}


def conv1d(p, x, stride=1):
    """x: (B, C, L) -> (B, C', L')"""
    y = jax.lax.conv_general_dilated(x, p["w"], (stride,), "SAME", dimension_numbers=("NCH", "OIH", "NCH"))
    return y + p["b"][None, :, None]


def layernorm(x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps)


# ---------------------------------------------------------------- modality encoders
def enc_init(key, kind, din, demb=64):
    k = jax.random.split(key, 6)
    if kind == "wave":                       # din = channels
        return {"c1": conv_init(k[0], din, 16, 7), "c2": conv_init(k[1], 16, 32, 5), "c3": conv_init(k[2], 32, 32, 5),
                "out": mlp_init(k[3], [64, 64, demb])}
    if kind == "seq":                        # short multichannel daily series
        return {"c1": conv_init(k[0], din, 16, 3), "out": mlp_init(k[1], [48, 64, demb])}
    return {"out": mlp_init(k[0], [din, 64, 64, demb])}     # tabular


def enc_apply(p, kind, x):
    if kind == "wave":
        h = jax.nn.gelu(conv1d(p["c1"], x, 2))
        h = jax.nn.gelu(conv1d(p["c2"], h, 2))
        h = jax.nn.gelu(conv1d(p["c3"], h, 2))
        h = jnp.concatenate([h.mean(-1), h.max(-1)], -1)
        return mlp(p["out"], h)
    if kind == "seq":
        h = jax.nn.gelu(conv1d(p["c1"], x, 1))
        h = jnp.concatenate([h.mean(-1), h.max(-1), jnp.sqrt(h.var(-1) + 1e-6)], -1)
        return mlp(p["out"], h)
    return mlp(p["out"], x)
