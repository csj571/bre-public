"""GP regression unit tests (numpy-only — no torch required).

The torch-dependent SomaticBayesianEngine tests live in
test_somatic_bayesian.py and skip cleanly when torch isn't installed.
"""
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from engine.gp import GaussianProcessRegressor


def test_engine_imports():
    assert GaussianProcessRegressor is not None


def test_cosine_kernels():
    # 2 sample inputs in 2D space
    X1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    X2 = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=float)

    # 1. Cosine RBF kernel test
    gp_rbf = GaussianProcessRegressor(kernel_type='cosine_rbf', ell=1.0, eta=1.0)
    k_val = gp_rbf._kernel(X1, X2)
    # k(X1[0], X2[0]): cos_sim=1.0, d_cos=0.0 -> value should be 1.0
    assert np.allclose(k_val[0, 0], 1.0)
    # k(X1[1], X2[0]): cos_sim=0.0, d_cos=1.0 -> value should be exp(-0.5) = 0.60653066
    assert np.allclose(k_val[1, 0], np.exp(-0.5))

    # 2. Cosine Matérn-5/2 kernel test
    gp_matern = GaussianProcessRegressor(kernel_type='cosine_matern52', ell=1.0, eta=1.0)
    k_val_mat = gp_matern._kernel(X1, X2)
    assert np.allclose(k_val_mat[0, 0], 1.0)
    # k(X1[1], X2[0]): d_cos=1.0 -> s = sqrt(5.0), value should be (1 + sqrt(5) + 5/3) * exp(-sqrt(5))
    s = np.sqrt(5.0)
    expected = (1.0 + s + 5.0 / 3.0) * np.exp(-s)
    assert np.allclose(k_val_mat[1, 0], expected)


def test_heteroscedastic_bald():
    gp = GaussianProcessRegressor(kernel_type='rbf', ell=1.0, eta=1.0, sigma_n=0.2)
    sig_f = np.array([0.4, 0.1, 0.9])

    # Base BALD score
    bald_base = gp.get_bald_score(sig_f)
    assert bald_base.shape[0] == 3

    # Heteroscedastic noise
    sig_n = np.array([0.1, 0.4, 0.05])
    bald_hetero = gp.get_bald_score(sig_f, sigma_n=sig_n)

    # Point 2 has higher sig_f/sig_n ratio (0.9/0.05 = 18) compared to point 0 (0.4/0.1 = 4)
    # so point 2's BALD score should be higher
    assert bald_hetero[2] > bald_hetero[0]


def test_gp_correlated_data():
    gp = GaussianProcessRegressor(kernel_type='rbf', ell=1.0, eta=1.0, sigma_n=1e-5)
    # Perfectly correlated (identical) data points to force singular matrix
    X = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=float)
    y = np.array([0.5, 0.5, 0.5], dtype=float)

    # This should not raise LinAlgError because of recursive jitter
    gp.fit(X, y)

    # Try optimize which also needs to survive singular matrices
    # It might not succeed (return True) due to flat gradients, but it shouldn't crash
    gp.optimize()


def test_gp_refit_refreshes_state():
    gp = GaussianProcessRegressor(kernel_type='matern52')
    X = np.linspace(0, 1, 10)[:, None]
    y = np.sin(X.ravel() * 6)
    gp.fit(X, y)
    m1, _ = gp.predict(np.array([[0.5]]))
    gp.fit(X, y + 0.5)
    m2, _ = gp.predict(np.array([[0.5]]))
    assert gp.L is not None
    assert abs(float(np.atleast_1d(m1)[0]) - float(np.atleast_1d(m2)[0])) > 1e-6
