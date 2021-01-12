from __future__ import print_function
from __future__ import division

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, nnls
from sklearn.utils import check_X_y
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.linear_model.base import LinearModel


def _rescale_data(X, y, sample_weight):
    """Rescale data so as to support sample_weight"""
    n_samples = X.shape[0]
    sample_weight = sample_weight * np.ones(n_samples)
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight, 0),
                                  shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y


class NonNegativeLinearRegression(LinearModel, RegressorMixin):
    """Non-negative least squares linear regression.

    Parameters
    ----------
    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        estimated coefficients more robust and almost independent of the
        number of samples. The same property is not valid for standardized
        data. However, if you wish to standardize, please use
        `preprocessing.StandardScaler` before calling `fit` on an estimator
        with `normalize=False`.
    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array, shape (n_features, )
        Estimated coefficients for the linear regression problem.

    intercept_ : array
        Independent term in the linear model.

    opt_result_ : OptimizeResult
        Result of non-negative least squares optimization
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

    def fit(self, X, y, sample_weight=None):
        """Fit non-negative linear model.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        y : numpy array of shape [n_samples,]
            Target values
        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample

        Returns
        -------
        self : returns an instance of self.

        """
        X, y = check_X_y(X, y, y_numeric=True, multi_output=False)

        if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
            raise ValueError("Sample weights must be 1D array or scalar")

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        self.coef_, result = nnls(X, y.squeeze())

        if np.all(self.coef_ == 0):
            raise ConvergenceWarning("All coefficients estimated to be zero in"
                                     " the non-negative least squares fit.")

        self._set_intercept(X_offset, y_offset, X_scale)
        self.opt_result_ = OptimizeResult(success=True, status=0, x=self.coef_,
                                          fun=result)
        return self
