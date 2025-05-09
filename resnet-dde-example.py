"""
Basic demonstration of Denoising Density Estimation (using a ResNet with softplus activations).
"""

import argparse
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import resnet_model as resffn

# from dataset_batcher import IterableDataset, twoway_random_split
# from toy_dataset_utils import create_dataset, create_eval_mesh, plot_grid
# from train_test_patterns import update_many_epochs

import matplotlib.pyplot as plt


def binary_ce(f, y):
    p = 1.0 / (1 + jnp.exp(-1 * f))
    return -1 * (y * jnp.log(p) + (1 - y) * jnp.log(1 - p))


@jax.jit
def loss(params, features, targets):
    preds = resffn.batched_predict(params, features)
    return jnp.mean(binary_ce(preds, targets))


@jax.jit
def update_wd(params, x, y, step_size, weight_decay):
    grads = jax.grad(loss)(params, x, y)
    return jax.tree_map(
        lambda p, dp: p - step_size * (dp + weight_decay * p), params, grads
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3, help="number of resnet layers")
    parser.add_argument("--units-per-layer", type=int, default=125)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--jax-seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--D", type=int, default=5)
    # parser.add_argument("--show-loss", action="store_true")
    # parser.add_argument("--show-function", action="store_true")
    parser.add_argument("--step-size", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    # parser.add_argument("--serve-jax", action="store_true")
    args = parser.parse_args()

    assert args.step_size > 0
    assert args.weight_decay >= 0

    layer_sizes = [args.units_per_layer for _ in range(args.layers + 1)]
    layer_sizes.insert(0, args.D)
    layer_sizes.append(1)

    print(layer_sizes)
    params = resffn.init_network_params(layer_sizes, jax.random.PRNGKey(args.jax_seed))

    print(
        "model has %i parameters, %i hidden layers, and takes %i inputs"
        % (resffn.num_parameters(params), len(params) - 2, args.D)
    )

    print(jax.tree_util.tree_structure(params))

    # TODO: the training for this example should simply be based on sampling from a unit Gaussian with dimension D
    # ...

    X = jnp.array(np.random.randn(*(args.N, args.D)))
    print(X.shape)

    fX = resffn.batched_predict_softplus(params, X)
    print(fX.shape)

    print(resffn.predict_softplus(params, X[0, :]))
    print(resffn.grad_predict_softplus(params, X[0, :]))

    gX = resffn.batched_grad_predict_softplus(params, X)
    print(gX.shape)

    """if args.show_loss:
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
        class0 = y.flatten() == 0
        plt.scatter(X[class0, 0], X[class0, 1], alpha=0.04, color="black")
        class1 = y.flatten() == 1
        plt.scatter(X[class1, 0], X[class1, 1], alpha=0.04, color="white")
        plt.show()"""

    print("done.")
