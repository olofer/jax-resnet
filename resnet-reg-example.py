"""
Demonstration of multivariate regression with the JAX resnet
"""

import argparse
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import resnet_model as resffn
from dataset_batcher import IterableDataset, twoway_random_split
from toy_dataset_utils import (
    create_dataset,
    create_eval_mesh,
    plot_grid,
    create_uniform_contrast,
)
from train_test_patterns import update_many_epochs
import matplotlib.pyplot as plt


@jax.jit
def mse_loss(params, features, targets):
    preds = resffn.batched_predict(params, features)
    error = targets - preds
    return 0.5 * jnp.mean(error * error)


@jax.jit
def mse_update_wd(params, x, y, step_size, weight_decay):
    grads = jax.grad(mse_loss)(params, x, y)
    return jax.tree_map(
        lambda p, dp: p - step_size * (dp + weight_decay * p), params, grads
    )


def make_test_dataset(N: int, M: int):
    X = np.random.randn(N)
    Y = np.column_stack([np.cos((k + 1) * 2 * np.pi * X / 4.0) for k in range(M)])
    return X.reshape((N, 1)), Y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3, help="number of resnet layers")
    parser.add_argument("--units-per-layer", type=int, default=125)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batches", type=int, default=15)
    parser.add_argument("--eprint", type=int, default=100)
    parser.add_argument("--bprint", type=int, default=0)
    parser.add_argument("--jax-seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=1500)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--show-loss", action="store_true")
    parser.add_argument("--show-function", action="store_true")
    parser.add_argument("--step-size", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--serve-jax", action="store_true")
    args = parser.parse_args()

    X, Y0 = make_test_dataset(args.N, args.M)
    Y = Y0 + np.random.randn(args.N, args.M) * args.sigma

    print([X.shape, Y.shape])

    layer_sizes = [args.units_per_layer for _ in range(args.layers + 1)]
    layer_sizes.insert(0, 1)  # single input
    layer_sizes.append(args.M)  # M outputs

    print(layer_sizes)
    params = resffn.init_network_params(layer_sizes, jax.random.PRNGKey(args.jax_seed))

    print(
        "model has %i parameters, and %i hidden layers"
        % (resffn.num_parameters(params), len(params) - 2)
    )

    assert args.step_size > 0
    assert args.weight_decay >= 0

    trainparams = {
        "bprint": args.bprint,
        "eprint": args.eprint,
        "step_size": args.step_size,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "shuffle": args.shuffle,
    }

    dataset1, dataset2 = twoway_random_split(X, Y, args.batches)

    if args.serve_jax:
        dataset1.serve_jax()
        dataset2.serve_jax()

    print("MSE cannot be lower than %f" % (0.5 * args.sigma**2))

    params, losses = update_many_epochs(
        params, dataset1, trainparams, mse_update_wd, mse_loss, dataset2
    )

    if args.show_loss:
        plt.plot(losses["train"], label="train set")
        plt.plot(losses["test"], label="test set")
        plt.xlabel("Epoch number")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    if args.show_function:
        Xhat = np.column_stack([np.linspace(-4.0, 4.0, 500)])
        Fhat = np.array(resffn.batched_predict(params, Xhat))

        idx = np.argsort(X, axis=0).flatten()

        for t in range(args.M):
            plt.scatter(X, Y[:, t], alpha=0.05, c="black", label="noisy samples")
            plt.plot(X[idx, 0], Y0[idx, t], c="green", label="True target", alpha=0.50)
            plt.plot(Xhat, Fhat[:, t], c="blue", label="ResNet output", alpha=0.50)
            plt.grid(True)
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y[%i]" % (t))
            plt.show()

    print("done.")
