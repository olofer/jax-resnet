"""
Basic demonstration of Denoising Density Estimation (using a ResNet with softplus activations).
Learn an unnormalized log-density function s(x). The basic scheme is enabled via Tweedie's formula.

Reference: https://doi.org/10.1109/TNNLS.2023.3308191
"""

import argparse
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import resnet_model as resffn
import matplotlib.pyplot as plt


@jax.jit
def loss(params, features, targets, sigmas):
    """
    targets: drawn from a unit normal distribution
    sigmas: sets the length scale of the smooth approximation of the density
    """
    grads = resffn.batched_grad_predict_softplus(params, features + sigmas * targets)
    err = grads.squeeze() + targets / sigmas
    mse_loss = jnp.mean(err * err)
    return mse_loss


@jax.jit
def update_wd(params, x, y, w, step_size, weight_decay):
    grads = jax.grad(loss)(params, x, y, w)
    return jax.tree_map(
        lambda p, dp: p - step_size * (dp + weight_decay * p), params, grads
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=5, help="number of resnet layers")
    parser.add_argument("--units-per-layer", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=25_000)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--jax-seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=10_000)
    parser.add_argument("--D", type=int, default=5)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()

    assert args.step_size > 0
    assert args.weight_decay >= 0

    assert args.sigma > 0, "sigma > 0 required"

    layer_sizes = [args.units_per_layer for _ in range(args.layers + 1)]
    layer_sizes.insert(0, args.D)
    layer_sizes.append(1)

    print(layer_sizes)
    params = resffn.init_network_params(layer_sizes, jax.random.PRNGKey(args.jax_seed))

    print(
        "model has %i parameters, %i hidden layers (%i units each), and takes D=%i inputs"
        % (resffn.num_parameters(params), len(params) - 2, args.units_per_layer, args.D)
    )

    print(jax.tree_util.tree_structure(params))

    # Smoke test som functions

    X = jnp.array(np.random.randn(*(args.N, args.D)))
    print(X.shape)

    fX_pre = resffn.batched_predict_softplus(params, X)
    print(fX_pre.shape)

    print(resffn.predict_softplus(params, X[0, :]))
    print(resffn.grad_predict_softplus(params, X[0, :]))

    gX = resffn.batched_grad_predict_softplus(params, X)
    print(gX.shape)

    # Sampling-based training loop

    print(
        "Sampling %i batches (each of size %i) with step-size=%f"
        % (args.num_batches, args.batch_size, args.step_size)
    )

    for b in range(args.num_batches):

        Xb = jnp.array(np.random.randn(*(args.batch_size, args.D)))  # features
        Ub = jnp.array(np.random.randn(*Xb.shape))  # targets
        Wb = jnp.array(np.tile(args.sigma, Xb.shape))

        loss_ = loss(params, Xb, Ub, Wb)
        print("batch %03i:" % (b), loss_)

        params = update_wd(params, Xb, Ub, Wb, args.step_size, args.weight_decay)

    # Evaluate the un-normalized log-density function

    fX_post = resffn.batched_predict_softplus(params, X)
    norm_sq_x = jnp.sum(X * X, axis=1)

    log_px = -0.5 * norm_sq_x - args.D * jnp.log(2 * jnp.pi) / 2
    assert len(log_px.shape) == 1

    log_unnormalized_integral = jax.scipy.special.logsumexp(
        fX_post.flatten() - log_px
    ) - jnp.log(X.shape[0])

    plt.plot(
        jnp.sqrt(norm_sq_x),
        fX_pre,
        linestyle="none",
        marker="s",
        alpha=0.10,
        color="orange",
        label="initial parameters",
    )
    plt.plot(
        jnp.sqrt(norm_sq_x),
        fX_post,
        linestyle="none",
        marker="o",
        alpha=0.10,
        color="blue",
        label="fitted (un-normalized)",
    )
    plt.plot(
        jnp.sqrt(norm_sq_x),
        fX_post - log_unnormalized_integral,
        linestyle="none",
        marker="o",
        alpha=0.10,
        color="green",
        label="fitted (normalized)",
    )
    plt.plot(
        jnp.sqrt(norm_sq_x),
        log_px,  # -0.5 * norm_sq_x,
        linestyle="none",
        marker=".",
        alpha=0.25,
        color="black",
        label="ideal/limit (normalized)",
    )
    plt.xlabel("$\|x\|_2$", fontsize=15)
    plt.ylabel("log-density $s(x) = \log p(x)$", fontsize=15)
    plt.title("Denoising Density Estimation (DDE) test D=%i" % (X.shape[1]))
    plt.legend()
    plt.grid(True)
    plt.show()

    print("done.")
