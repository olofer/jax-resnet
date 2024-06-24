import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


def create_eval_mesh(xrange=(-4.0, 4.0), yrange=(-3.0, 3.0), ngrid=int(200)):
    x = np.linspace(xrange[0], xrange[1], ngrid)
    y = np.linspace(yrange[0], yrange[1], ngrid)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv.flatten(), yv.flatten()], axis=-1)


def plot_grid(x12, f12, cmap="bwr", fcn: callable = None, title_str: str = None):
    assert x12.shape[1] == 2
    ngrid = int(np.sqrt(x12.shape[0]))
    assert ngrid**2 == x12.shape[0]
    x1g = np.linspace(np.min(x12[:, 0]), np.max(x12[:, 0]), ngrid)
    x2g = np.linspace(np.min(x12[:, 1]), np.max(x12[:, 1]), ngrid)
    bbox = (x1g[0], x1g[-1], x2g[0], x2g[-1])
    p12 = 1.0 / (1.0 + np.exp(-1 * f12)) if fcn is None else fcn(f12)
    plt.imshow(p12.reshape((ngrid, ngrid)), origin="lower", extent=bbox, cmap=cmap)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.colorbar()
    if not title_str is None:
        plt.title(title_str)


def create_dataset(
    n: int, noise_level: float = 0.10, typestr: str = "moons", label_invert=False
):
    if typestr == "circles":
        X, y = sklearn.datasets.make_circles(
            n_samples=n, noise=noise_level, factor=0.67
        )
    elif typestr == "moons":
        X, y = sklearn.datasets.make_moons(n_samples=n, noise=noise_level)
    elif typestr == "ripple":
        # X = np.random.randn(n, 2)
        X = (2 * np.random.rand(n, 2) - 1) * 3
        r = np.sqrt(np.sum(X * X, axis=1))
        prob = np.cos(2 * np.pi * r / 4) ** 2
        y = np.array([1 if np.random.rand() < p else 0 for p in prob])
    else:
        raise NotImplementedError

    if label_invert:
        y = 1 - y

    return X, np.array(y.reshape(X.shape[0], 1), dtype=np.float64)
