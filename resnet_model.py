"""
ResNet with Layer Normalization in JAX
"""

import jax
import jax.numpy as jnp
import numpy as np


def random_layer_params(m, n, key, scale=None, uniform=True):
    w_key, b_key = jax.random.split(key)
    if scale is None:
        scale = 1.0 / jnp.sqrt(m)  # m is the input size
    if uniform:
        weights = 2 * scale * (jax.random.uniform(w_key, (n, m)) - 0.5)
        biases = 2 * scale * (jax.random.uniform(b_key, (n,)) - 0.5)
    else:
        weights = scale * jax.random.normal(w_key, (n, m))
        biases = scale * jax.random.normal(b_key, (n,))
    return {
        "weight": weights,
        "bias": biases,
    }


def init_network_params(sizes, key):
    keys = jax.random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def num_parameters(params):
    return np.sum(
        [np.prod(l["weight"].shape) + np.prod(l["bias"].shape) for l in params]
    )


def relu(x):
    return jnp.maximum(0, x)


def predict(params, x):
    LN_EPS = 1.0e-6
    activations = jnp.dot(params[0]["weight"], x) + params[0]["bias"]
    for p in params[1:-1]:
        mean = jnp.mean(activations)
        var = jnp.var(activations)
        activations = (activations - mean) / jnp.sqrt(var + LN_EPS)
        outputs = jnp.dot(p["weight"], activations) + p["bias"]
        activations = relu(outputs) + outputs

    logits = jnp.dot(params[-1]["weight"], activations) + params[-1]["bias"]
    return logits


batched_predict = jax.vmap(predict, in_axes=(None, 0))
