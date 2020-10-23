from __future__ import print_function
from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np
import six
import warnings

from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.model_selection import check_cv
from sklearn.utils import tosequence, check_X_y

try:
    # TODO: Avoid using a private function from scikit-learn.
    #  _check_fit_params was added at sklearn 0.22.1
    from sklearn.utils.validation import _check_fit_params
except ImportError:
    # _index_param_value was removed in sklearn 0.22.1
    # See: https://github.com/scikit-learn/scikit-learn/pull/15863
    from sklearn.model_selection._validation import _index_param_value

    warnings.warn(
        'Your civisml-extensions installation uses private functions from '
        'scikit-learn < v0.22.1. Please upgrade scikit-learn to v0.22.1 '
        'or beyond. A future version of civisml-extensions will no longer '
        'be compatible with scikit-learn < v0.22.1.',
        FutureWarning
    )

    def _check_fit_params(X, fit_params, train):
        return {k: _index_param_value(X, v, train)
                for k, v in fit_params.items()}


def _fit_est(est, X, y, **fit_params):
    return est.fit(X, y, **fit_params)


def _reshape_2d_long(arr):
    # Reshape output so it's always 2-d and long
    if arr.ndim < 2:
        arr = arr.reshape(-1, 1)
    return arr


def _regressor_predict(est, X):
    """standardized predictions for regression"""
    return _reshape_2d_long(est.predict(X))


def _regressor_fit_predict(est, Xtrn, ytrn, Xtst, **fit_params):
    """function for doing fit and predict for regressors"""
    est.fit(Xtrn, ytrn, **fit_params)
    return _regressor_predict(est, Xtst)


def _classifier_predict(est, X):
    # Note: this prefers a decision_function to predict_proba
    # when both are present (e.g. logistic regression), per the
    # convention in CalibratedClassifierCV.
    if hasattr(est, "decision_function"):
        ypred = est.decision_function(X)
    elif hasattr(est, "predict_proba"):
        # predict_proba rows always sum to 1, so drop last col
        ypred = est.predict_proba(X)[:, :-1]
    elif hasattr(est, "predict"):
        # you may want to allow predict for pass-through estimators
        ypred = est.predict(X)
    else:
        raise RuntimeError("Estimator without a `decision_function`, "
                           "`predict_proba`, or `predict` method supplied to a"
                           " StackedClassifier.")

    return _reshape_2d_long(ypred)


def _classifier_fit_predict(est, Xtrn, ytrn, Xtst, **fit_params):
    """function for doing fit and predict for classifiers"""
    est.fit(Xtrn, ytrn, **fit_params)
    return _classifier_predict(est, Xtst)


@six.add_metaclass(ABCMeta)
class BaseStackedModel(BaseEstimator):
    """Abstract base class for StackedClassifier and StackedRegressor. It is
    loosely based on sklearn.pipeline.Pipeline.
    """
    def __init__(self,
                 estimator_list,
                 cv=3,
                 n_jobs=1,
                 pre_dispatch='2*n_jobs',
                 verbose=0):
        self.estimator_list = tosequence(estimator_list)
        self.cv = cv
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.verbose = verbose

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: boolean, optional (default: True)
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        out : mapping of string to any
            Parameter names mapped to their values.
        """
        # If not deep, just get the params of the BaseStackedModel object
        out = super(BaseStackedModel, self).get_params(deep=False)
        if not deep:
            return out
        # If deep, extract parameters from estimators too
        est_list = getattr(self, 'estimator_list')
        # If est_list is an empty list, don't do anything else
        if len(est_list) > 0:
            out.update(self.named_base_estimators.copy())
            out.update({self.meta_estimator_name: self.meta_estimator}.copy())
            for name, estimator in est_list:
                for key, value in estimator.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        # Ensure strict ordering of parameter setting:
        # 1. All estimators
        if 'estimator_list' in params:
            setattr(self, 'estimator_list', params.pop('estimator_list'))
        # 2. Estimator replacement
        est_names, _ = zip(*getattr(self, 'estimator_list'))
        for name in list(params.keys()):
            if '__' not in name and name in est_names:
                self._replace_est('estimator_list', name, params.pop(name))
        # 3. Estimator parameters and other initilisation arguments
        super(BaseStackedModel, self).set_params(**params)

        return self

    def fit(self, X, y, **fit_params):
        """Fit the model

        Fit the base estimators on CV folds, then use their prediction on the
        validation folds to train the meta-estimator. Then re-fit base
        estimators on full training set.

        Parameters
        ----------
        X : np.ndarray, list of numbers
            Training data.
        y : np.ndarray, list of numbers
            Training targets.
        **fit_params : dict of {string, object}
            Parameters passed to the ``fit`` method of each estimator, where
            each parameter name is prefixed such that parameter ``p`` for
            estimator ``s`` has key ``s__p``.

        Returns
        -------
        self : BaseStackedModel
            This estimator
        """
        self._validate_estimators()
        X, y = check_X_y(X, y, multi_output=True)

        # Fit base estimators on CV training folds, produce features for
        # meta-estimator from predictions on CV test folds.
        Xmeta, ymeta, meta_params = self._base_est_fit_predict(X, y,
                                                               **fit_params)
        # Fit meta-estimator on test fold predictions of base estimators.
        self.meta_estimator.fit(Xmeta, ymeta, **meta_params)
        # Now fit base estimators again, this time on full training set
        self._base_est_fit(X, y, **fit_params)

        return self

    # _replace_est copied nearly verbatim from sklearn.pipeline._BasePipeline
    # v0.18.1 "_replace_step" method.
    def _replace_est(self, ests_attr, name, new_val):
        # assumes `name` is a valid est name
        new_ests = getattr(self, ests_attr)[:]
        for i, (est_name, _) in enumerate(new_ests):
            if est_name == name:
                new_ests[i] = (name, new_val)
                break
        setattr(self, ests_attr, new_ests)

    # _validate_names copied nearly verbatim from
    # sklearn.pipeline._BasePipeline v0.18.1
    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))

    @abstractmethod
    def _validate_estimators(self):
        pass

    def _extract_fit_params(self, **fit_params):
        """Extract fit parameters for each estimator and store in nested dict
        """
        fit_params_ests = dict((name, {}) for name, est in self.estimator_list)
        for pname, pval in fit_params.items():
            est, param = pname.split('__', 1)
            fit_params_ests[est][param] = pval

        return fit_params_ests

    def _base_est_fit(self, X, y, **fit_params):
        """Fit the base estimators on X and y.
        """
        fit_params_ests = self._extract_fit_params(**fit_params)

        _jobs = []
        for name, est in self.estimator_list[:-1]:
            _jobs.append(delayed(_fit_est)(
                clone(est), X, y, **fit_params_ests[name]))

        _out = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch)(_jobs)

        for name, _ in self.estimator_list[:-1]:
            self._replace_est('estimator_list', name, _out.pop(0))

    @abstractmethod
    def _check_cv(self, y):
        pass

    def _base_est_fit_predict(self, X, y, **fit_params):
        """Fit the base estimators on CV training folds, and return their
        out-of-sample predictions on the test folds as features for the
        meta-estimator. Also return the fit_params for the meta-estimator.
        """
        y = y.squeeze()
        # Construct CV iterator
        cv = self._check_cv(y=y)
        # Extract CV indices since we need them twice, and un-seeded CV
        # generators with `shuffle=True` split differently each time.
        train_inds = []
        test_inds = []
        for train, test in cv.split(X, y):
            train_inds.append(train)
            test_inds.append(test)

        fit_params_ests = self._extract_fit_params(**fit_params)
        _fit_predict = self._get_fit_predict_function()

        _jobs = []

        # Loop over CV folds to get out-of-sample predictions, which become the
        # features for the meta-estimator.
        for train, test in zip(train_inds, test_inds):
            for name, est in self.estimator_list[:-1]:
                # adapted from sklearn.model_selection._fit_and_predict
                # Adjust length of sample weights
                fit_params_est_adjusted = _check_fit_params(
                    X, fit_params_ests[name], train
                )

                # Fit estimator on training set and score out-of-sample
                _jobs.append(delayed(_fit_predict)(
                    clone(est),
                    X[train],
                    y[train],
                    X[test],
                    **fit_params_est_adjusted))

        _out = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch)(_jobs)

        # Extract the results from joblib
        Xmeta, ymeta = None, None
        for test in test_inds:
            ybase = np.empty((y[test].shape[0], 0))
            for name, est in self.estimator_list[:-1]:
                # Build design matrix out of out-of-sample predictions
                ybase = np.hstack((ybase, _out.pop(0)))

            # Append the test outputs to what will eventually be the features
            # for the meta-estimator.
            if Xmeta is not None:
                ymeta = np.concatenate((ymeta, y[test]))
                Xmeta = np.vstack((Xmeta, ybase))
            else:
                Xmeta = ybase
                ymeta = y[test]

        return Xmeta, ymeta, fit_params_ests[self.meta_estimator_name]

    def _base_est_predict(self, X):
        """Return base estimator predictions on X as meta-features.
        """
        Xmeta = np.empty((X.shape[0], 0))
        for name, est in self.estimator_list[:-1]:
            Xmeta = np.hstack((Xmeta, self._est_predict(est, X)))

        return Xmeta

    @abstractmethod
    def _est_predict(self, est, X):
        """This function ensures that the relevant prediction function is
        consistently called for a given base estimator.
        """
        pass

    @abstractmethod
    def _get_fit_predict_function(self):
        """return the function to be used for the fit and out-of-sample
        predictions"""
        pass

    @if_delegate_has_method(delegate='meta_estimator')
    def predict(self, X):
        """Run predictions through base estimators, then predict with the
        meta-estimator on the output of the base estimators.

        Parameters
        ----------
        X : np.ndarray, list of numbers
            Data to predict on.

        Returns
        -------
        y_pred : array-like
        """
        Xmeta = self._base_est_predict(X)

        return self.meta_estimator.predict(Xmeta)

    @if_delegate_has_method(delegate='meta_estimator')
    def score(self, X, y, **params):
        """Run predictions through base estimators, then score with the
        meta-estimator on the output of the base estimators.

        Parameters
        ----------
        X : np.ndarray, list of numbers
            Data to predict on.

        y : np.ndarray, list of numbers
            Targets for scoring. Must fulfill label requirements for
            meta-estimator.

        params: dict of {string, object}
            Parameters passed to the ``score`` method of the meta-estimator.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xmeta = self._base_est_predict(X)

        return self.meta_estimator.score(Xmeta, y, **params)

    @property
    def named_base_estimators(self):
        return dict(self.estimator_list[:-1])

    @property
    def meta_estimator(self):
        return self.estimator_list[-1][-1]

    @property
    def meta_estimator_name(self):
        return self.estimator_list[-1][0]


class StackedClassifier(BaseStackedModel):
    """Builds a stacked classification model from a list of named estimators,
    using the final estimator in the list as the meta-estimator.

    This class takes a list of named estimators, and it fits all but the last
    of these estimators (called the "base estimators") on part of the training
    data passed to ``fit``. The remaining training data is used to create
    out-of-sample predictions from these base estimators. The final named
    estimator, called the meta-estimator, is trained on these out-of-sample
    predictions. This allows the meta-estimator to optimally aggregate the
    predictions of several base estimators, hopefully improving on their
    individual predictive powers.

    It is loosely based on sklearn.pipeline.Pipeline.

    Parameters
    ----------
    estimator_list: list of (str, estimator) tuples
        This contains tuples holding the name and estimator of the desired base
        and meta-estimators. The meta-estimator MUST be the final item in
        the list. The order of the base estimators is irrelevant, as long as
        they occur before the meta-estimator.
    cv : int, cross-validation generator, or iterable, optional (default: 3)
        Determines the cross-validation splitting strategy. Possible inputs for
        cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    n_jobs : int, (default: 1)
        Number of jobs to run in parallel.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    verbose : integer
        Controls the verbosity: the higher, the more messages. A value
        of 10 gives a moderate level of logging. 50 or more is the most
        amount of logging.

    Attributes
    ----------
    named_base_estimators : dict
        Read-only attribute to access any base estimator by user-given name.
        Keys are estimator names and values are estimators.
    meta_estimator: estimator
        The meta-estimator, provided as a separately accessible property.

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from civismlext.stacking import StackedClassifier
    >>> # Note that the final estimator 'metalr' is the meta-estimator
    >>> estlist = [('rf', RandomForestClassifier()),
    >>>            ('lr', LogisticRegression()),
    >>>            ('metalr', LogisticRegression())]
    >>> mysm = StackedClassifier(estlist)
    >>> # Set some parameters, if you didn't set them at instantiation
    >>> mysm.set_params(rf__random_state=7, lr__random_state=8,
    >>>                 metalr__random_state=9, metalr__C=10**7)
    >>> # Fit
    >>> mysm.fit(Xtrain, ytrain)
    >>> # Predict!
    >>> ypred = mysm.predict_proba(Xtest)
    """
    def __init__(self,
                 estimator_list,
                 cv=3,
                 n_jobs=1,
                 pre_dispatch='2*n_jobs',
                 verbose=0):
        super(StackedClassifier, self).__init__(
            estimator_list, cv, n_jobs, pre_dispatch, verbose)

    def _validate_estimators(self):
        """Validates that the names and methods of the estimators match
        expectiations. Overrides `validate_estimators` in `BaseStackedModel`.
        """
        names, estimators = zip(*self.estimator_list)

        if len(self.estimator_list) < 2:
            raise RuntimeError("You must have two or more estimators to fit a "
                               "StackedClassifier!")

        # validate names
        self._validate_names(names)

        # validate meta-estimator
        if not hasattr(self.meta_estimator, "fit"):
            raise TypeError("Meta-estimator '%s' does not have fit method." %
                            self.meta_estimator_name)
        _check_classifier_methods(self.meta_estimator,
                                  self.meta_estimator_name)

        # validate base estimators
        for name, est in self.estimator_list[:-1]:
            if not hasattr(est, "fit"):
                raise TypeError("Estimator '%s' does not have fit method." %
                                name)
            _check_classifier_methods(est, name)

    def _check_cv(self, y):
        """Overrides base class _check_cv
        """
        # Squeezed target should be 1-dimensional
        if len(y.shape) != 1:
            raise NotImplementedError("StackedClassifier does not currently "
                                      "support multi-column classification "
                                      "problems. If your target is a one-hot "
                                      "encoded multi-class problem, please "
                                      "recast it to a single column.")
        return check_cv(self.cv, y=y, classifier=True)

    def _est_predict(self, est, X):
        """This function ensures that the relevant prediction function is
        consistently called for a given base estimator.
        """
        return _classifier_predict(est, X)

    def _get_fit_predict_function(self):
        """return the function to be used for the fit and out-of-sample
        predictions"""
        return _classifier_fit_predict

    @if_delegate_has_method(delegate='meta_estimator')
    def predict_proba(self, X):
        """Run predictions through base estimators, then predict class
        probabilities with the meta-estimator on the output of the base
        estimators.

        Parameters
        ----------
        X : np.ndarray, list of numbers
            Data to predict on.

        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        Xmeta = self._base_est_predict(X)

        return self.meta_estimator.predict_proba(Xmeta)

    @if_delegate_has_method(delegate='meta_estimator')
    def decision_function(self, X):
        """Run predictions through base estimators, then pass the output of the
        base estimators to the meta-estimator's decision_function.

        Parameters
        ----------
        X : np.ndarray, list of numbers
            Data to predict on.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xmeta = self._base_est_predict(X)

        return self.meta_estimator.decision_function(Xmeta)

    @if_delegate_has_method(delegate='meta_estimator')
    def predict_log_proba(self, X):
        """Run predictions through base estimators, then predict class log
        probabilities with the meta-estimator on the output of the base
        estimators.

        Parameters
        ----------
        X : np.ndarray, list of numbers
            Data to predict on.

        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xmeta = self._base_est_predict(X)

        return self.meta_estimator.predict_log_proba(Xmeta)

    @property
    def classes_(self):
        return self.meta_estimator.classes_


def _check_classifier_methods(clf, name):
    """Checks whether clf has either a `predict_proba`, `decision_function`, or
    `predict` method. Raises if none of these methods are present.

    Parameters
    ----------
    clf: estimator
        Estimator to check. Expected to be a classifier.

    name: str
        Name of estimator, to be output in case estimator is missing expected
        classification functionality.
    """
    if not (hasattr(clf, "predict_proba") or hasattr(clf, "predict") or
            hasattr(clf, "decision_function")):
        raise RuntimeError("Estimator '%s' does not have `predict_prob`, "
                           "`decision_function`, or `predict` method." % name)


class StackedRegressor(BaseStackedModel):
    """Builds a stacked regression model from a list of named estimators, using
    the final estimator in the list as the meta-estimator.

    This class takes a list of named estimators, and it fits all but the last
    of these estimators (called the "base estimators") on part of the training
    data passed to ``fit``. The remaining training data is used to create
    out-of-sample predictions from these base estimators. The final named
    estimator, called the meta-estimator, is trained on these out-of-sample
    predictions. This allows the meta-estimator to optimally aggregate the
    predictions of several base estimators, hopefully improving on their
    individual predictive powers.

    It is loosely based on sklearn.pipeline.Pipeline.

    Parameters
    ----------
    estimator_list: list of (str, estimator) tuples
        This contains tuples holding the name and estimator of the desired base
        and meta-estimators. The meta-estimator MUST be the final item in
        the list. The order of the base estimators is irrelevant, as long as
        they occur before the meta-estimator.
    cv : int, cross-validation generator, or iterable, optional (default: 3)
        Determines the cross-validation splitting strategy. Possible inputs for
        cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        :class:`sklearn.model_selection.KFold` is used by default for
        regression targets.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    n_jobs : int, (default: 1)
        Number of jobs to run in parallel.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    verbose : integer
        Controls the verbosity: the higher, the more messages. A value
        of 10 gives a moderate level of logging. 50 or more is the most
        amount of logging.

    Attributes
    ----------
    named_base_estimators : dict
        Read-only attribute to access any base estimator by user-given name.
        Keys are estimator names and values are estimators.
    meta_estimator: estimator
        The meta-estimator, provided as a separately accessible property.

    Example
    -------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from civismlext.stacking import StackedRegressor
    >>> from civismlext.nonnegative import NonNegativeLinearRegression
    >>> # Note that the final estimator 'meta_nnr' is the meta-estimator
    >>> estlist = [('rf', RandomForestRegressor()),
    >>>            ('lr', LinearRegression()),
    >>>            ('meta_nnr', NonNegativeLinearRegression())]
    >>> mysm = StackedRegressor(estlist)
    >>> # Set some parameters, if you didn't set them at instantiation
    >>> mysm.set_params(rf__random_state=7)
    >>> # Fit
    >>> mysm.fit(Xtrain, ytrain)
    >>> # Predict!
    >>> ypred = mysm.predict(Xtest)
    """
    def __init__(self,
                 estimator_list,
                 cv=3,
                 n_jobs=1,
                 pre_dispatch='2*n_jobs',
                 verbose=0):
        super(StackedRegressor, self).__init__(
            estimator_list, cv, n_jobs, pre_dispatch, verbose)

    def _validate_estimators(self):
        """Validates that the names and methods of the estimators match
        expectations. Overrides `validate_estimators` in `BaseStackedModel`.
        """
        names, estimators = zip(*self.estimator_list)

        if len(self.estimator_list) < 2:
            raise RuntimeError("You must have two or more estimators to fit a "
                               "StackedRegressor!")

        # validate names
        self._validate_names(names)

        # validate meta-estimator
        if not hasattr(self.meta_estimator, "fit"):
            raise TypeError("Meta-estimator '%s' does not have fit method." %
                            self.meta_estimator_name)
        # validate base estimators
        for name, est in self.estimator_list[:-1]:
            if not hasattr(est, "fit"):
                raise TypeError("Estimator '%s' does not have fit method." %
                                name)

    def _check_cv(self, y):
        """Overrides base class _check_cv
        """
        return check_cv(self.cv, y=y, classifier=False)

    def _est_predict(self, est, X):
        """This function ensures that the relevant prediction function is
        consistently called for a given base estimator.
        """
        return _regressor_predict(est, X)

    def _get_fit_predict_function(self):
        """return the function to be used for the fit and out-of-sample
        predictions"""
        return _regressor_fit_predict
