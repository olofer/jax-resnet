"""
Multi-class probability regression using the JAX ResNet.

Synthetic problem data generated from a simple CTMC (one state).
There are K-1 different paths to jump out of the state and each jump has its own rate.
Each delta-time K things can happen, either remain in the state or jump out.
The jump rates are all different functions of a observed state vector x.
The state x moves via a stochastic process until a jump even occurs (or timed out).
Multiclass logits can be fitted the usual way, and then the jump rates can be backed out.

SEE: https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html

EXAMPLE:
  python3 resnet-mce-example.py  --show-loss --show-function

"""

import argparse
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import resnet_model as resffn
from dataset_batcher import IterableDataset, twoway_random_split
from toy_dataset_utils import (
    create_eval_mesh,
    plot_grid,
)
from train_test_patterns import update_many_epochs
import matplotlib.pyplot as plt


# Each row of f = multiclass logits, with 0-th element the "no event" class
def rates_from_logits(f: np.ndarray, dt: float = 1.0):
    f0 = f[:, 0].reshape((f.shape[0], 1))
    C = -1.0 * f0 / (1 - np.exp(f0)) / dt
    return np.exp(f[:, 1:]) * np.repeat(C, f.shape[1] - 1, axis=1)


def numpy_one_hot(x, k, dtype=np.float32):
    return np.array(x[:, None] == np.arange(k), dtype)


@jax.jit
def mce_loss(params, features, targets):
    logprobs = resffn.batched_predict_multi_logits(params, features)
    return -1.0 * jnp.mean(targets * logprobs)


@jax.jit
def mce_update_wd(params, x, y, step_size, weight_decay):
    grads = jax.grad(mce_loss)(params, x, y)
    return jax.tree_map(
        lambda p, dp: p - step_size * (dp + weight_decay * p), params, grads
    )


def gillespie_sample_path(
    x0: np.ndarray,
    drift: callable,
    diffusion: callable,
    jump_rates: callable,
    deltatime: float = 0.1,
    timeout: float = 100.0,
) -> dict:
    assert deltatime > 0
    sqrt_dt = np.sqrt(deltatime)
    kmax = int(np.ceil(timeout / deltatime))
    x = np.copy(x0).flatten()
    D = len(x)
    X = np.tile(np.nan, (kmax, D))
    k, t, event = int(0), float(0.0), False
    while not event and k < kmax:
        X[k, :] = x
        rates = jump_rates(t, x)
        assert np.all(rates > 0), "any (listed) jump rate must be positive"
        total_hazard = np.sum(rates)
        event = np.random.rand() < 1 - np.exp(-1.0 * deltatime * total_hazard)
        which_event = (
            int(0)
            if not event
            else 1 + np.random.choice(len(rates), p=rates / total_hazard)
        )
        x += drift(t, x) * deltatime + diffusion(t, x) * sqrt_dt * np.random.randn(D)
        t += deltatime
        k += 1

    return {"X": X[:k, :], "event": which_event}


def generate_gillespie_dataset(
    D: int, N: int, mu: callable, sigma: callable, rates: callable, **kwargs
):
    def expanded_target_(l: int, event: int) -> np.array:
        y = np.zeros((l, 1), dtype=np.int32)
        y[l - 1, 0] = event
        return y

    path_count = int(0)
    n = int(0)
    X = None
    Y = None
    while n < N:
        path = gillespie_sample_path(
            np.zeros(D),
            drift=mu,
            diffusion=sigma,
            jump_rates=rates,
            deltatime=kwargs["deltatime"],
            timeout=kwargs["timeout"],
        )
        X = np.row_stack([X, path["X"]]) if X is not None else path["X"]
        Y = (
            np.row_stack([Y, expanded_target_(path["X"].shape[0], path["event"])])
            if Y is not None
            else expanded_target_(path["X"].shape[0], path["event"])
        )
        n += path["X"].shape[0]
        assert n == X.shape[0]
        assert X.shape[1] == D
        assert len(Y) == n
        path_count += 1

    return X, Y, path_count


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3, help="number of resnet layers")
    parser.add_argument("--units-per-layer", type=int, default=125)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batches", type=int, default=15)
    parser.add_argument("--eprint", type=int, default=100)
    parser.add_argument("--bprint", type=int, default=0)
    parser.add_argument("--jax-seed", type=int, default=42)
    parser.add_argument("--N", type=int, default=20000)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--show-loss", action="store_true")
    parser.add_argument("--show-function", action="store_true")
    parser.add_argument("--eval-function", action="store_true")
    parser.add_argument("--which-function", type=str, default="A")
    parser.add_argument("--step-size", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--serve-jax", action="store_true")
    args = parser.parse_args()

    DT = 0.05
    TIMEOUT = 10.0

    def mufunc(t, x):
        return np.zeros(x.shape)

    def sigmafunc(t, x):
        return np.ones(x.shape)

    if args.which_function == "A":

        def hazardfunc(t, x):
            return np.array([0.10, 0.02, 0.20])

    elif args.which_function == "B":

        def hazardfunc(t, x):
            R = np.sqrt(np.sum(x * x))
            return np.array(
                [0.10 + 0.10 * np.sin(R) ** 2, 0.02 + 0.05 * R**2, 0.20 + 0.05 * R]
            )

    else:
        raise ValueError("--which-function option value not recognized")
    
    print("employing hazard function %s" % (args.which_function))

    print("generating data (requesting at least %i samples).." % (args.N))
    X, Y, paths_ = generate_gillespie_dataset(
        int(2), args.N, mufunc, sigmafunc, hazardfunc, deltatime=DT, timeout=TIMEOUT
    )

    print("sampled from %i paths" % (paths_))
    seen_event_types = np.unique(Y)
    print([X.shape, Y.shape])

    num_possible_events = len(hazardfunc(0.0, np.array([0.0, 0.0]))) + 1
    print(
        "seen event types: %i, possible types: %i"
        % (len(seen_event_types), num_possible_events)
    )

    Y = numpy_one_hot(Y.flatten(), num_possible_events)
    print(Y.shape)

    layer_sizes = [args.units_per_layer for _ in range(args.layers + 1)]
    layer_sizes.insert(0, 2)  # state space dim. = 2
    layer_sizes.append(num_possible_events)  # K-class output -> logits vector

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

    params, losses = update_many_epochs(
        params, dataset1, trainparams, mce_update_wd, mce_loss, dataset2
    )

    if args.eval_function:
        Yhat = resffn.batched_predict_multi_logits(params, X)
        Rhat = rates_from_logits(Yhat, dt=DT)
        Rtru = np.row_stack([hazardfunc(None, X[k, :]) for k in range(X.shape[0])])
        assert np.all(Rhat.shape == Rtru.shape)

        print("mean estimated jump rates (along paths):")
        print(np.mean(np.array(Rhat), axis=0))
        print("mean true jump rates (along paths):")
        print(np.mean(np.array(Rtru), axis=0))
        print("mean absolute jump rate error (along paths):")
        print(np.mean(np.abs(np.array(Rhat - Rtru)), axis=0))

    if args.show_loss:
        plt.plot(losses["train"], label="train set")
        plt.plot(losses["test"], label="test set")
        plt.xlabel("Epoch number")
        plt.ylabel("MCE Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    if args.show_function:
        x1_range = np.quantile(np.abs(X[:, 0]), 0.90)
        x2_range = np.quantile(np.abs(X[:, 1]), 0.90)

        X12 = create_eval_mesh(
            xrange=(-x1_range, x1_range), yrange=(-x2_range, x2_range)
        )
        Y12 = resffn.batched_predict_multi_logits(params, X12)
        R12 = rates_from_logits(Y12, dt=DT)

        print(X12.shape)
        print(Y12.shape)
        print(R12.shape)

        for c in range(R12.shape[1]):
            plt.figure(figsize=(10, 6))
            plot_grid(
                X12,
                R12[:, c],
                title_str="$\lambda_{%i}$" % (c + 1),
                fcn=lambda x: x,
                cmap="viridis",
            )
            plt.scatter(x=X[:, 0], y=X[:, 1], alpha=0.05, c="black", s=1.0)
            plt.gca().set_xlim((-x1_range, x1_range))
            plt.gca().set_ylim((-x2_range, x2_range))
            plt.show()

    print("done.")
