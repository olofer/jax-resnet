"""
Density estimation via the "likelihood ratio trick" using a JAX ResNet
"""

import argparse
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import resnet_model as resffn
from dataset_batcher import IterableDataset, random_split
from toy_dataset_utils import create_dataset, create_eval_mesh, plot_grid
from train_test_patterns import update_many_epochs
import matplotlib.pyplot as plt

# TODO: define loss functions etc
# TODO: define the augmented training data with a uniform contrast or a Gaussian contrast (argparser)

if __name__ == "__main__":
    print("done.")
