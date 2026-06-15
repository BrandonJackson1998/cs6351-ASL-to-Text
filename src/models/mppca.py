"""Mixture of Probabilistic PCA (Tipping & Bishop 1999, §3).

A K-component MPPCA models data as

    p(x) = Σ_k π_k * N(x; μ_k, W_k W_k^T + σ²_k I)

where each component is a PPCA: low-rank Gaussian with mean μ_k, loading
matrix W_k ∈ R^(D×q), and isotropic noise σ²_k. Fit via EM:

  E-step:  posterior responsibilities r_{n,k} = π_k * N(x_n; μ_k, C_k) / Σ_j (...)
  M-step:  refit π_k, μ_k, W_k, σ²_k from responsibility-weighted statistics

The classifier wraps one MPPCA per class (letter). Classification picks the
class whose mixture log-likelihood is highest.
"""

import os
import string
import pickle
from typing import Optional, Tuple

import numpy as np

from src.models.ppca import PPCA


_LETTERS = list(string.ascii_uppercase)
_EPS = 1e-12


class MPPCA:
    """Mixture of PPCAs fit via EM. One instance models one class."""

    def __init__(
        self,
        n_components: int = 3,
        latent_dim: int = 12,
        max_iter: int = 50,
        tol: float = 1e-3,
        seed: int = 0,
    ):
        self.K = n_components
        self.q = latent_dim
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.pis_ = None       # (K,) mixing weights
        self.mus_ = None       # (K, D) component means
        self.Ws_ = None        # list of (D, q) loadings
        self.sigma2s_ = None   # (K,) per-component noise variance
        self.D_ = None         # input dim

    # ----------------------------------------------------------------------
    # Per-component log-likelihood (Woodbury for efficient inverse)
    # ----------------------------------------------------------------------

    def _log_pdf(self, X, k):
        """Return log N(x; μ_k, W_k W_k^T + σ²_k I), shape (n,)."""
        D = self.D_
        q = self.q
        Xc = X - self.mus_[k]
        sigma2 = max(self.sigma2s_[k], _EPS)
        W = self.Ws_[k]

        WtW = W.T @ W
        A = sigma2 * np.eye(q) + WtW
        sign, logdet_A = np.linalg.slogdet(A)
        log_det_C = (D - q) * np.log(sigma2) + logdet_A

        A_inv = np.linalg.inv(A)
        WtXc = Xc @ W
        quad = (
            np.einsum("nd,nd->n", Xc, Xc)
            - np.einsum("nq,qp,np->n", WtXc, A_inv, WtXc)
        ) / sigma2

        return -0.5 * (D * np.log(2 * np.pi) + log_det_C + quad)

    def log_likelihood(self, X) -> np.ndarray:
        """Log p(x) under the mixture, shape (n,)."""
        X = np.asarray(X, dtype=np.float64)
        # Per-component log-pdfs + log priors; then logsumexp
        comp = np.zeros((X.shape[0], self.K), dtype=np.float64)
        for k in range(self.K):
            comp[:, k] = self._log_pdf(X, k) + np.log(self.pis_[k] + _EPS)
        m = comp.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(comp - m).sum(axis=1, keepdims=True))).ravel()

    # ----------------------------------------------------------------------
    # EM training
    # ----------------------------------------------------------------------

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, D = X.shape
        K, q = self.K, self.q
        if q >= D:
            raise ValueError("latent_dim must be < n_features")
        self.D_ = D

        rng = np.random.default_rng(self.seed)
        # Initialize: random component assignment, then refit each component
        # as a PPCA on its assigned points
        labels = rng.integers(0, K, size=n)
        # Ensure no empty component
        for k in range(K):
            if (labels == k).sum() < q + 2:
                # Force-assign a few random points to keep component non-empty
                idx = rng.choice(n, size=max(q + 2, 5), replace=False)
                labels[idx] = k

        self.pis_ = np.full(K, 1.0 / K, dtype=np.float64)
        self.mus_ = np.zeros((K, D), dtype=np.float64)
        self.Ws_ = [np.zeros((D, q), dtype=np.float64) for _ in range(K)]
        self.sigma2s_ = np.full(K, 1.0, dtype=np.float64)

        for k in range(K):
            mask = labels == k
            pts = X[mask]
            if len(pts) < q + 2:
                pts = X[: q + 2]
            self._refit_component(k, pts, weights=None)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            # E-step
            log_comp = np.zeros((n, K), dtype=np.float64)
            for k in range(K):
                log_comp[:, k] = self._log_pdf(X, k) + np.log(self.pis_[k] + _EPS)
            m = log_comp.max(axis=1, keepdims=True)
            log_norm = (m + np.log(np.exp(log_comp - m).sum(axis=1, keepdims=True))).ravel()
            log_R = log_comp - log_norm[:, None]
            R = np.exp(log_R)  # (n, K), responsibilities

            ll = float(log_norm.sum())
            if abs(ll - prev_ll) < self.tol * max(1.0, abs(prev_ll)):
                break
            prev_ll = ll

            # M-step
            Nk = R.sum(axis=0) + _EPS  # (K,)
            self.pis_ = Nk / n
            for k in range(K):
                self._refit_component(k, X, weights=R[:, k])

        return self

    def _refit_component(self, k, X, weights=None):
        """Closed-form PPCA fit of component k from (weighted) data."""
        D, q = self.D_, self.q
        if weights is None:
            mu = X.mean(axis=0)
            Xc = X - mu
            S = (Xc.T @ Xc) / max(len(X), 1)
        else:
            w = weights
            wsum = w.sum() + _EPS
            mu = (w[:, None] * X).sum(axis=0) / wsum
            Xc = X - mu
            S = (Xc * w[:, None]).T @ Xc / wsum

        eigvals, eigvecs = np.linalg.eigh(S)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        top_vals = eigvals[:q]
        top_vecs = eigvecs[:, :q]
        tail = eigvals[q:]
        sigma2 = float(max(tail.mean() if len(tail) else _EPS, _EPS))

        diag = np.maximum(top_vals - sigma2, 0.0)
        W = top_vecs * np.sqrt(diag)

        self.mus_[k] = mu
        self.Ws_[k] = W
        self.sigma2s_[k] = sigma2

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "K": self.K, "q": self.q,
                    "pis": self.pis_, "mus": self.mus_,
                    "Ws": self.Ws_, "sigma2s": self.sigma2s_,
                    "D": self.D_,
                },
                f,
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(n_components=d["K"], latent_dim=d["q"])
        obj.pis_ = d["pis"]; obj.mus_ = d["mus"]
        obj.Ws_ = d["Ws"]; obj.sigma2s_ = d["sigma2s"]
        obj.D_ = d["D"]
        return obj


class MPPCAClassifier:
    """One MPPCA per letter. Classify by maximum a-posteriori under the mixtures.

    Equivalent to PPCAMixtureClassifier but with K components per class.
    """

    def __init__(
        self,
        n_components_per_class: int = 3,
        latent_dim: int = 12,
        max_iter: int = 50,
        seed: int = 0,
    ):
        self.K = n_components_per_class
        self.q = latent_dim
        self.max_iter = max_iter
        self.seed = seed
        self.scaler = None
        self.classes_ = None
        self.log_priors_ = None
        self.mppcas_ = {}

    def fit(self, X, y):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.classes_ = np.unique(y)
        n_total = len(y)
        self.log_priors_ = {}
        for cls in self.classes_:
            mask = y == cls
            pts = Xs[mask]
            print(f"  fitting class {int(cls)} ({mask.sum()} samples)...", flush=True)
            self.mppcas_[int(cls)] = MPPCA(
                n_components=self.K, latent_dim=self.q,
                max_iter=self.max_iter, seed=self.seed + int(cls),
            ).fit(pts)
            self.log_priors_[int(cls)] = float(np.log(mask.sum() / n_total + _EPS))
        return self

    def _log_posteriors(self, X):
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float64))
        n = Xs.shape[0]
        out = np.zeros((n, len(self.classes_)), dtype=np.float64)
        for j, cls in enumerate(self.classes_):
            out[:, j] = self.mppcas_[int(cls)].log_likelihood(Xs) + self.log_priors_[int(cls)]
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self._log_posteriors(X), axis=1)]

    def predict_proba(self, X):
        lp = self._log_posteriors(X)
        lp -= lp.max(axis=1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        for cls, m in self.mppcas_.items():
            m.save(os.path.join(dir_path, f"mppca_class_{cls}.pkl"))
        np.savez(
            os.path.join(dir_path, "scaler.npz"),
            mean=self.scaler.mean_, scale=self.scaler.scale_,
            var=self.scaler.var_, n_features=self.scaler.n_features_in_,
        )
        with open(os.path.join(dir_path, "config.pkl"), "wb") as f:
            pickle.dump(
                {
                    "K": self.K, "q": self.q,
                    "classes": [int(c) for c in self.classes_],
                    "log_priors": self.log_priors_,
                },
                f,
            )

    @classmethod
    def load(cls, dir_path):
        with open(os.path.join(dir_path, "config.pkl"), "rb") as f:
            cfg = pickle.load(f)
        obj = cls(n_components_per_class=cfg["K"], latent_dim=cfg["q"])
        obj.classes_ = np.array(cfg["classes"])
        obj.log_priors_ = {int(k): float(v) for k, v in cfg["log_priors"].items()}
        for c in obj.classes_:
            obj.mppcas_[int(c)] = MPPCA.load(os.path.join(dir_path, f"mppca_class_{int(c)}.pkl"))
        from sklearn.preprocessing import StandardScaler
        sd = np.load(os.path.join(dir_path, "scaler.npz"))
        obj.scaler = StandardScaler()
        obj.scaler.mean_ = sd["mean"]; obj.scaler.scale_ = sd["scale"]
        obj.scaler.var_ = sd["var"]
        obj.scaler.n_features_in_ = int(sd["n_features"])
        return obj
