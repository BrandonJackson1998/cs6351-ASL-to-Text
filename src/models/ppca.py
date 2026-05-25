"""
Probabilistic Principal Component Analysis (Tipping & Bishop, 1999).

Closed-form fit via eigendecomposition of the sample covariance matrix.
Provides transform (latent posterior mean), inverse_transform (reconstruction),
and log_likelihood for use as a generative classifier.
"""

import numpy as np


class PPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.W_ = None
        self.sigma2_ = None
        self.M_inv_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        q = self.n_components
        if q >= d:
            raise ValueError(f"n_components ({q}) must be < n_features ({d})")

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        S = (Xc.T @ Xc) / n

        eigvals, eigvecs = np.linalg.eigh(S)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        top_vals = eigvals[:q]
        top_vecs = eigvecs[:, :q]
        tail = eigvals[q:]
        sigma2 = max(tail.mean(), 1e-10) if len(tail) > 0 else 1e-10

        diag = np.maximum(top_vals - sigma2, 0.0)
        self.W_ = top_vecs * np.sqrt(diag)
        self.sigma2_ = float(sigma2)

        M = self.W_.T @ self.W_ + sigma2 * np.eye(q)
        self.M_inv_ = np.linalg.inv(M)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mean_
        return Xc @ self.W_ @ self.M_inv_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.W_.T + self.mean_

    def log_likelihood(self, X):
        """Log-likelihood under PPCA model: x ~ N(mean, W W^T + sigma2 I).

        Uses Woodbury identity to avoid forming d x d covariance matrix.
        Returns shape (n,) array of per-sample log-likelihoods.
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        q = self.n_components
        Xc = X - self.mean_

        sigma2 = self.sigma2_
        W = self.W_

        WtW = W.T @ W
        A = sigma2 * np.eye(q) + WtW
        sign, logdet_A = np.linalg.slogdet(A)
        # log|C| = log|sigma2 I_d + W W^T| = (d - q) log sigma2 + log|sigma2 I_q + W^T W|
        log_det_C = (d - q) * np.log(sigma2) + logdet_A

        # C^{-1} = (1/sigma2) (I - W A^{-1} W^T)
        A_inv = np.linalg.inv(A)
        WtXc = Xc @ W
        quad = (np.einsum("nd,nd->n", Xc, Xc) - np.einsum("nq,qp,np->n", WtXc, A_inv, WtXc)) / sigma2

        return -0.5 * (d * np.log(2 * np.pi) + log_det_C + quad)

    def save(self, path):
        np.savez(
            path,
            mean=self.mean_,
            W=self.W_,
            sigma2=np.array(self.sigma2_),
            n_components=np.array(self.n_components),
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        obj = cls(int(data["n_components"]))
        obj.mean_ = data["mean"]
        obj.W_ = data["W"]
        obj.sigma2_ = float(data["sigma2"])
        q = obj.n_components
        M = obj.W_.T @ obj.W_ + obj.sigma2_ * np.eye(q)
        obj.M_inv_ = np.linalg.inv(M)
        return obj
