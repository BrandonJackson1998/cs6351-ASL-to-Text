"""Classifiers built on top of PPCA.

PPCALogisticClassifier: standardize -> PPCA -> logistic regression.
PPCAMixtureClassifier: one PPCA per class; classify by max log-likelihood + log prior.
"""

import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.ppca import PPCA


class PPCALogisticClassifier:
    def __init__(self, n_components=12, C=1.0, max_iter=1000):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.ppca = PPCA(n_components)
        self.clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        Z = self.ppca.fit_transform(Xs)
        self.clf.fit(Z, y)
        return self

    def transform(self, X):
        Xs = self.scaler.transform(X)
        return self.ppca.transform(Xs)

    def predict(self, X):
        return self.clf.predict(self.transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.transform(X))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        self.ppca.save(os.path.join(dir_path, "ppca.npz"))
        np.savez(
            os.path.join(dir_path, "scaler.npz"),
            mean=self.scaler.mean_,
            scale=self.scaler.scale_,
            var=self.scaler.var_,
            n_features=self.scaler.n_features_in_,
        )
        with open(os.path.join(dir_path, "classifier.pkl"), "wb") as f:
            pickle.dump(self.clf, f)
        with open(os.path.join(dir_path, "config.json"), "w") as f:
            json.dump({"type": "ppca_logistic", "n_components": self.n_components}, f)

    @classmethod
    def load(cls, dir_path):
        with open(os.path.join(dir_path, "config.json")) as f:
            cfg = json.load(f)
        obj = cls(n_components=cfg["n_components"])
        obj.ppca = PPCA.load(os.path.join(dir_path, "ppca.npz"))
        sd = np.load(os.path.join(dir_path, "scaler.npz"))
        obj.scaler = StandardScaler()
        obj.scaler.mean_ = sd["mean"]
        obj.scaler.scale_ = sd["scale"]
        obj.scaler.var_ = sd["var"]
        obj.scaler.n_features_in_ = int(sd["n_features"])
        with open(os.path.join(dir_path, "classifier.pkl"), "rb") as f:
            obj.clf = pickle.load(f)
        return obj


class PPCAMixtureClassifier:
    """Generative classifier: one PPCA per class, classify by max posterior.

    log p(y|x) ∝ log p(x|y) + log p(y)
    """

    def __init__(self, n_components=12):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.ppcas_ = {}
        self.log_priors_ = {}
        self.classes_ = None

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.classes_ = np.unique(y)
        n_total = len(y)
        for cls in self.classes_:
            mask = y == cls
            self.ppcas_[int(cls)] = PPCA(self.n_components).fit(Xs[mask])
            self.log_priors_[int(cls)] = float(np.log(mask.sum() / n_total))
        return self

    def _log_posteriors(self, X):
        Xs = self.scaler.transform(X)
        n = Xs.shape[0]
        log_post = np.zeros((n, len(self.classes_)))
        for j, cls in enumerate(self.classes_):
            log_post[:, j] = self.ppcas_[int(cls)].log_likelihood(Xs) + self.log_priors_[int(cls)]
        return log_post

    def predict(self, X):
        log_post = self._log_posteriors(X)
        return self.classes_[np.argmax(log_post, axis=1)]

    def predict_proba(self, X):
        log_post = self._log_posteriors(X)
        log_post -= log_post.max(axis=1, keepdims=True)
        p = np.exp(log_post)
        return p / p.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        for cls, model in self.ppcas_.items():
            model.save(os.path.join(dir_path, f"ppca_class_{cls}.npz"))
        np.savez(
            os.path.join(dir_path, "scaler.npz"),
            mean=self.scaler.mean_,
            scale=self.scaler.scale_,
            var=self.scaler.var_,
            n_features=self.scaler.n_features_in_,
        )
        with open(os.path.join(dir_path, "config.json"), "w") as f:
            json.dump(
                {
                    "type": "ppca_mixture",
                    "n_components": self.n_components,
                    "classes": [int(c) for c in self.classes_],
                    "log_priors": {str(k): v for k, v in self.log_priors_.items()},
                },
                f,
            )

    @classmethod
    def load(cls, dir_path):
        with open(os.path.join(dir_path, "config.json")) as f:
            cfg = json.load(f)
        obj = cls(n_components=cfg["n_components"])
        obj.classes_ = np.array(cfg["classes"])
        obj.log_priors_ = {int(k): float(v) for k, v in cfg["log_priors"].items()}
        for c in obj.classes_:
            obj.ppcas_[int(c)] = PPCA.load(os.path.join(dir_path, f"ppca_class_{int(c)}.npz"))
        sd = np.load(os.path.join(dir_path, "scaler.npz"))
        obj.scaler = StandardScaler()
        obj.scaler.mean_ = sd["mean"]
        obj.scaler.scale_ = sd["scale"]
        obj.scaler.var_ = sd["var"]
        obj.scaler.n_features_in_ = int(sd["n_features"])
        return obj
