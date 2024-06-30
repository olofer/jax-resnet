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
def update_wd(params, x, y, step_size, weight_decay):
    grads = jax.grad(mse_loss)(params, x, y)
    return jax.tree_map(
        lambda p, dp: p - step_size * (dp + weight_decay * p), params, grads
    )


if __name__ == "__main__":

    #
    # TODO: scalar feature X, multiple targets Y(X) + noise, ResNet in between... go!
    #

    print("done.")
