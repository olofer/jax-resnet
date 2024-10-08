import numpy as np
import jax.numpy as jnp


class IterableDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, batches: int, w: np.ndarray = None):
        assert len(X.shape) == 2
        assert len(y.shape) == 2
        assert batches >= 1
        assert X.shape[0] == y.shape[0]
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.w = None if w is None else np.copy(w)
        if not self.w is None:
            assert len(w.shape) == 2
            assert w.shape[0] == y.shape[0]
        tmp = np.linspace(0, 1, batches + 1)
        self.splits = [int(np.round(k * self.X.shape[0])) for k in tmp]
        self.serve_jnp = False

    def __len__(self):
        return len(self.splits) - 1

    def __getitem__(self, idx):
        irange = np.arange(self.splits[idx], self.splits[idx + 1])
        if self.serve_jnp:
            if self.w is None:
                return jnp.array(self.X[irange, :]), jnp.array(self.y[irange, :])
            else:
                return (
                    jnp.array(self.X[irange, :]),
                    jnp.array(self.y[irange, :]),
                    jnp.array(self.w[irange, :]),
                )
        else:
            if self.w is None:
                return self.X[irange, :], self.y[irange, :]
            else:
                return self.X[irange, :], self.y[irange, :], self.w[irange, :]

    def size(self):
        return self.X.shape[0]
    
    def has_weights(self):
        return not self.w is None

    def shuffle(self):
        idx = np.random.permutation(self.X.shape[0])
        self.X = self.X[idx, :]
        self.y = self.y[idx, :]
        if not self.w is None:
            self.w = self.w[idx, :]

    def serve_numpy(self):
        self.serve_jnp = False

    def serve_jax(self):
        self.serve_jnp = True


def twoway_random_split(X: np.ndarray, y: np.ndarray, B: int, weights: np.ndarray = None):
    def local_slice(A: np.ndarray, idx):
        assert len(A.shape) == 1 or len(A.shape) == 2
        return A[idx].reshape((len(idx), 1)) if len(A.shape) == 1 else A[idx, :]

    idxshuffle = np.random.permutation(X.shape[0])
    idx1 = idxshuffle[: (X.shape[0] // 2)]
    idx2 = idxshuffle[(X.shape[0] // 2) :]

    d1 = IterableDataset(
        local_slice(X, idx1),
        local_slice(y, idx1),
        batches=B,
        w=None if weights is None else local_slice(weights, idx1),
    )
    d2 = IterableDataset(
        local_slice(X, idx2),
        local_slice(y, idx2),
        batches=B,
        w=None if weights is None else local_slice(weights, idx2),
    )

    return d1, d2
