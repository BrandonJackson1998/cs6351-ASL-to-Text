"""Unit tests for PPCA on synthetic data."""

import numpy as np
import pytest

from src.models.ppca import PPCA


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n, d, q = 500, 10, 3
    W_true = rng.standard_normal((d, q))
    Z = rng.standard_normal((n, q))
    noise = rng.standard_normal((n, d)) * 0.1
    X = Z @ W_true.T + noise
    return X, q


def test_fit_recovers_subspace_dimension(synthetic_data):
    X, q = synthetic_data
    model = PPCA(n_components=q).fit(X)
    assert model.W_.shape == (X.shape[1], q)
    assert model.sigma2_ > 0
    assert model.sigma2_ < 0.5


def test_transform_inverse_reconstruction(synthetic_data):
    X, q = synthetic_data
    model = PPCA(n_components=q).fit(X)
    Z = model.transform(X)
    X_rec = model.inverse_transform(Z)
    err = np.mean((X - X_rec) ** 2)
    assert err < 0.5


def test_log_likelihood_shape_and_finite(synthetic_data):
    X, q = synthetic_data
    model = PPCA(n_components=q).fit(X)
    ll = model.log_likelihood(X)
    assert ll.shape == (X.shape[0],)
    assert np.all(np.isfinite(ll))


def test_save_load_roundtrip(synthetic_data, tmp_path):
    X, q = synthetic_data
    model = PPCA(n_components=q).fit(X)
    path = tmp_path / "ppca.npz"
    model.save(str(path))
    loaded = PPCA.load(str(path))
    np.testing.assert_allclose(loaded.transform(X), model.transform(X), rtol=1e-6)
    np.testing.assert_allclose(loaded.log_likelihood(X), model.log_likelihood(X), rtol=1e-6)
