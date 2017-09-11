from __future__ import print_function
from __future__ import division

import pytest
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from civismlext.nonnegative import NonNegativeLinearRegression
from civismlext.nonnegative import _rescale_data

rng = np.random.RandomState(17)


def test_rescale_data():
    """Copied from sklearn/linear_model/tests/test_base.py
    """
    n_samples = 200
    n_features = 2

    sample_weight = 1.0 + rng.rand(n_samples)
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    rescaled_X, rescaled_y = _rescale_data(X, y, sample_weight)
    rescaled_X2 = X * np.sqrt(sample_weight)[:, np.newaxis]
    rescaled_y2 = y * np.sqrt(sample_weight)
    np.testing.assert_allclose(rescaled_X, rescaled_X2)
    np.testing.assert_allclose(rescaled_y, rescaled_y2)


def test_nonneg_linreg():
    X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])

    nnlr = NonNegativeLinearRegression(fit_intercept=True)
    nnlr.fit(X, X.dot([1, 2]) + 2)

    np.testing.assert_allclose(nnlr.coef_, [1, 2])
    np.testing.assert_allclose(nnlr.intercept_, [2])
    np.testing.assert_allclose(nnlr.predict(np.array([[1, 1]])), [5])


def test_zero_coef_exception():
    with pytest.raises(ConvergenceWarning):
        X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])

        nnlr = NonNegativeLinearRegression(fit_intercept=True)
        nnlr.fit(X, X.dot([-1, -1]) + -1)


def test_preprocess_data():
    """Copied from sklearn/linear_model/tests/test_base.py, with small
    modifications.
    """
    n_samples = 200
    n_features = 2
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    expected_X_mean = np.mean(X, axis=0)
    expected_X_norm = np.std(X, axis=0) * np.sqrt(X.shape[0])
    expected_y_mean = np.mean(y, axis=0)
    nnlr = NonNegativeLinearRegression()

    Xt, yt, X_mean, y_mean, X_norm = \
        nnlr._preprocess_data(X, y, fit_intercept=False, normalize=False)
    np.testing.assert_allclose(X_mean, np.zeros(n_features))
    np.testing.assert_allclose(y_mean, 0)
    np.testing.assert_allclose(X_norm, np.ones(n_features))
    np.testing.assert_allclose(Xt, X)
    np.testing.assert_allclose(yt, y)

    Xt, yt, X_mean, y_mean, X_norm = \
        nnlr._preprocess_data(X, y, fit_intercept=True, normalize=False)
    np.testing.assert_allclose(X_mean, expected_X_mean)
    np.testing.assert_allclose(y_mean, expected_y_mean)
    np.testing.assert_allclose(X_norm, np.ones(n_features))
    np.testing.assert_allclose(Xt, X - expected_X_mean)
    np.testing.assert_allclose(yt, y - expected_y_mean)

    Xt, yt, X_mean, y_mean, X_norm = \
        nnlr._preprocess_data(X, y, fit_intercept=True, normalize=True)
    np.testing.assert_allclose(X_mean, expected_X_mean)
    np.testing.assert_allclose(y_mean, expected_y_mean)
    np.testing.assert_allclose(X_norm, expected_X_norm)
    np.testing.assert_allclose(Xt, (X - expected_X_mean) / expected_X_norm)
    np.testing.assert_allclose(yt, y - expected_y_mean)


def test_preprocess_data_weighted():
    """Copied from sklearn/linear_model/tests/test_base.py, with small
    modifications.
    """
    n_samples = 200
    n_features = 2
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    sample_weight = rng.rand(n_samples)
    expected_X_mean = np.average(X, axis=0, weights=sample_weight)
    expected_y_mean = np.average(y, axis=0, weights=sample_weight)
    nnlr = NonNegativeLinearRegression()

    # XXX: if normalize=True, should we expect a weighted standard deviation?
    #      Currently not weighted, but calculated with respect to weighted mean
    expected_X_norm = (np.sqrt(X.shape[0]) *
                       np.mean((X - expected_X_mean) ** 2, axis=0) ** .5)

    Xt, yt, X_mean, y_mean, X_norm = \
        nnlr._preprocess_data(X, y, fit_intercept=True, normalize=False,
                              sample_weight=sample_weight)
    np.testing.assert_allclose(X_mean, expected_X_mean)
    np.testing.assert_allclose(y_mean, expected_y_mean)
    np.testing.assert_allclose(X_norm, np.ones(n_features))
    np.testing.assert_allclose(Xt, X - expected_X_mean)
    np.testing.assert_allclose(yt, y - expected_y_mean)

    Xt, yt, X_mean, y_mean, X_norm = \
        nnlr._preprocess_data(X, y, fit_intercept=True, normalize=True,
                              sample_weight=sample_weight)
    np.testing.assert_allclose(X_mean, expected_X_mean)
    np.testing.assert_allclose(y_mean, expected_y_mean)
    np.testing.assert_allclose(X_norm, expected_X_norm)
    np.testing.assert_allclose(Xt, (X - expected_X_mean) / expected_X_norm)
    np.testing.assert_allclose(yt, y - expected_y_mean)
