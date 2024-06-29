"""
Density estimation via the "likelihood ratio trick" using a JAX ResNet
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


def binary_ce(f, y):
    p = 1.0 / (1 + jnp.exp(-1 * f))
    return -1 * (y * jnp.log(p) + (1 - y) * jnp.log(1 - p))


@jax.jit
def weighted_loss(params, features, targets, weights):
    preds = resffn.batched_predict(params, features)
    return jnp.mean(binary_ce(preds, targets) * weights)


@jax.jit
def weighted_update_wd(params, x, y, w, step_size, weight_decay):
    grads = jax.grad(weighted_loss)(params, x, y, w)
    return jax.tree_map(
        lambda p, dp: p - step_size * (dp + weight_decay * p), params, grads
    )


def stack_contrast_datasets(
    X0: np.array, X1: np.array, W0: float = None, W1: float = None
):
    N0 = X0.shape[0]
    N1 = X1.shape[0]

    X = np.row_stack([X0, X1])
    Y = np.row_stack([np.zeros((N0, 1)), np.ones((N1, 1))])

    # These automatic weights follow Hastie et al ESL section 14.2.4
    if W0 is None:
        W0 = N1 / (N1 + N0)
    if W1 is None:
        W1 = N0 / (N1 + N0)

    W = np.row_stack([np.tile(W0, (N0, 1)), np.tile(W1, (N1, 1))])
    return X, Y, W


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
    parser.add_argument("--N0", type=int, default=1500)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--show-loss", action="store_true")
    parser.add_argument("--show-function", action="store_true")
    parser.add_argument("--step-size", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--serve-jax", action="store_true")
    args = parser.parse_args()

    X1, _ = create_dataset(args.N, typestr="moons")
    X0, AB = create_uniform_contrast(X1, args.N0)
    X, Y, W = stack_contrast_datasets(X0, X1, W0=1.0, W1=1.0)
    print([X.shape, Y.shape, W.shape])

    layer_sizes = [args.units_per_layer for _ in range(args.layers + 1)]
    layer_sizes.insert(0, 2)
    layer_sizes.append(1)

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

    dataset1, dataset2 = twoway_random_split(X, Y, args.batches, weights=W)

    if args.serve_jax:
        dataset1.serve_jax()
        dataset2.serve_jax()

    assert dataset1.has_weights() and dataset2.has_weights()

    params, losses = update_many_epochs(
        params, dataset1, trainparams, weighted_update_wd, weighted_loss, dataset2
    )

    if args.show_loss:
        plt.plot(losses["train"], label="train set")
        plt.plot(losses["test"], label="test set")
        plt.xlabel("Epoch number")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    if args.show_function:
        X12 = create_eval_mesh()
        F12 = np.array(resffn.batched_predict(params, X12))
        plt.figure(figsize=(10, 6))
        plot_grid(X12, F12, title_str="probability")
        # class0 = y.flatten() == 0
        # plt.scatter(X[class0, 0], X[class0, 1], alpha=0.04, color="black")
        # class1 = y.flatten() == 1
        # plt.scatter(X[class1, 0], X[class1, 1], alpha=0.04, color="white")
        plt.show()

    print("done.")
