import numpy as np
import jax.numpy as jnp


class IterableDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, batches: int):
        assert len(X.shape) == 2
        assert len(y.shape) == 2
        assert batches >= 1
        assert X.shape[0] == y.shape[0]
        self.X = np.copy(X)
        self.y = np.copy(y)
        tmp = np.linspace(0, 1, batches + 1)
        self.splits = [int(np.round(k * self.X.shape[0])) for k in tmp]
        self.serve_jnp = False

    def __len__(self):
        return len(self.splits) - 1

    def __getitem__(self, idx):
        irange = np.arange(self.splits[idx], self.splits[idx + 1])
        if self.serve_jnp:
            return jnp.array(self.X[irange, :]), jnp.array(self.y[irange, :])
        else:
            return self.X[irange, :], self.y[irange, :]

    def size(self):
        return self.X.shape[0]

    def shuffle(self):
        idx = np.random.permutation(self.X.shape[0])
        self.X = self.X[idx, :]
        self.y = self.y[idx, :]

    def serve_numpy(self):
        self.serve_jnp = False

    def serve_jax(self):
        self.serve_jnp = True


def random_split(X: np.array, y: np.array, B: int):
    def local_slice(A: np.array, idx):
        assert len(A.shape) == 1 or len(A.shape) == 2
        return A[idx].reshape((len(idx), 1)) if len(A.shape) == 1 else A[idx, :]

    idxshuffle = np.random.permutation(X.shape[0])
    idx1 = idxshuffle[: (X.shape[0] // 2)]
    idx2 = idxshuffle[(X.shape[0] // 2) :]

    d1 = IterableDataset(local_slice(X, idx1), local_slice(y, idx1), batches=B)
    d2 = IterableDataset(local_slice(X, idx2), local_slice(y, idx2), batches=B)

    return d1, d2
