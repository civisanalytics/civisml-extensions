from __future__ import print_function
from __future__ import division

from .. import stacking
from ..stacking import StackedRegressor, StackedClassifier
from ..nonnegative import NonNegativeLinearRegression

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_regression, make_classification
from numpy.testing import assert_almost_equal, assert_array_equal
import six

if six.PY2:
    import mock
else:
    from unittest import mock


def generate_data(slope=1, magnitude=2, sigma=0.1,
                  rng=np.random.RandomState(7), ntrain=50, ntest=1000,
                  classification=False):
    totsize = ntrain + ntest
    X = rng.normal(scale=5, size=(totsize, 2))

    # Set yint = 0 if even, yint = 1 if odd
    yints = magnitude*(X[:, 0].round() % 2) - magnitude/2

    if classification:
        probs = 1/(1 + np.exp(-(slope*X[:, 0] + 0.1*X[:, 0]*X[:, 1] + yints)))
        y = rng.binomial(1, probs)
    else:
        y = slope*X[:, 0] + yints + rng.normal(scale=sigma, size=totsize)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ntest,
                                                        random_state=rng)
    return X_train.reshape(-1, 2), X_test.reshape(-1, 2), y_train, y_test


@pytest.fixture
def clf_test_data():
    rng = np.random.RandomState(7)
    xtrain, xtest, ytrain, ytest = generate_data(slope=0.5,
                                                 classification=True,
                                                 rng=rng)
    return {"x": xtrain, "y": ytrain, "xtest": xtest, "ytest": ytest}


@pytest.fixture
def regression_test_data():
    rng = np.random.RandomState(7)
    xtrain, xtest, ytrain, ytest = generate_data(rng=rng)
    return {"x": xtrain, "y": ytrain, "xtest": xtest, "ytest": ytest}


class NoFit(object):
    """Small class to test parameter dispatching.
    """
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class Transformer(NoFit):
    """Simple transformer-like class
    """
    def fit(self, X, y, **fit_params):
        self.Xfit = X
        self.yfit = y
        self.fit_params = fit_params
        return self

    def get_params(self, deep=False):
        return {'a': self.a, 'b': self.b}

    def set_params(self, **params):
        self.a = params['a']
        return self

    def transform(self, X):
        return X


class BasicEst(NoFit):
    """Simple estimator-like class
    """
    def fit(self, X, y, **fit_params):
        self.Xfit = X
        self.yfit = y
        self.fit_params = fit_params
        return self

    def get_params(self, deep=False):
        return {'a': self.a, 'b': self.b}

    def set_params(self, **params):
        self.a = params['a']
        return self

    def predict(self, X):
        self.Xpred = X
        return np.array([[3, 4],
                         [5, 6],
                         [7, 8]])

    def score(self, X, y):
        pass


class ClfEst(BasicEst):
    """Classifier-like class, with classes_
    """
    def fit(self, X, y, **fit_params):
        super(ClfEst, self).fit(X, y, **fit_params)
        self.classes_ = [0, 1]
        return self

    def predict_proba(self, X):
        self.Xpred = X
        return np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])


class FullClfEst(ClfEst):
    """Classifier-like class, with decision_function
    """
    def decision_function(self, X):
        self.Xpred = X
        return np.array([1, 2, 3])


class PassThruClf(ClfEst):
    def predict(self, X):
        raise RuntimeError("you shouldn't be here")

    def predict_proba(self, X):
        return self.Xfit


class PassThruDF(ClfEst):
    def predict(self, X):
        raise RuntimeError("you shouldn't be here")

    def predict_proba(self, X):
        raise RuntimeError("you shouldn't be here")

    def decision_function(self, X):
        return self.Xfit


class PassThruReg(ClfEst):
    def predict(self, X):
        return self.Xfit

    def predict_proba(self, X):
        raise RuntimeError("you shouldn't be here")

    def decision_function(self, X):
        raise RuntimeError("you shouldn't be here")


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_init(SM):
    """Construct some StackedModels and check their parameters
    """
    with pytest.raises(TypeError):
        sm = SM()

    # Smoke test with estimator-like object
    basic_est = BasicEst()
    sm = SM([('be', basic_est)], cv=9)  # NOQA


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_get_params(SM):
    basic_est = BasicEst()
    meta_est = BasicEst()
    est_list = [('be', basic_est), ('me', meta_est)]

    sm = SM(
        est_list, cv=9, n_jobs=10, pre_dispatch=1, verbose=10)

    # Check parameter getting
    assert sm.get_params(deep=False) == dict(cv=9,
                                             estimator_list=est_list,
                                             n_jobs=10,
                                             pre_dispatch=1,
                                             verbose=10)

    assert sm.get_params(deep=True) == dict(be__a=None, be__b=None,
                                            be=basic_est, me__a=None,
                                            me__b=None, me=meta_est,
                                            **sm.get_params(deep=False))


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_get_params_estimator_list_empty(SM):
    est_list = []

    sm = SM(
        est_list, cv=9, n_jobs=10, pre_dispatch=1, verbose=10)

    # Check parameter getting
    param_dict = dict(cv=9,
                      estimator_list=est_list,
                      n_jobs=10,
                      pre_dispatch=1,
                      verbose=10)
    assert sm.get_params(deep=False) == param_dict
    assert sm.get_params(deep=True) == param_dict


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_set_params(SM):
    basic_est = BasicEst()
    meta_est = BasicEst()
    est_list = [('be', basic_est), ('me', meta_est)]

    sm = SM(
        est_list, cv=9, n_jobs=10, pre_dispatch=1, verbose=10)

    # Check parameter setting
    sm.set_params(be__a=17)
    assert basic_est.a == 17
    assert sm.get_params(deep=True) == dict(be__a=17, be__b=None,
                                            be=basic_est, me__a=None,
                                            me__b=None, me=meta_est,
                                            **sm.get_params(deep=False))

    # Wrong names should raise exception
    with pytest.raises(ValueError):
        sm.set_params(foo__param=2)


def test_properties_clf():
    """Test that properties get set as expected.
    """
    base_clf = ClfEst(3, 5)
    meta_clf = ClfEst(4, 6)
    sm = StackedClassifier([('be', base_clf), ('me', meta_clf)])

    assert sm.meta_estimator == meta_clf
    assert sm.meta_estimator_name == 'me'
    assert sm.named_base_estimators == {'be': base_clf}

    with pytest.raises(AttributeError):
        sm.classes_
    meta_clf.fit('foo', 'bar')
    assert sm.classes_ == [0, 1]


def test_properties_regression():
    """Test that properties get set as expected.
    """
    base_reg = BasicEst(5, 7)
    meta_reg = BasicEst(6, 8)
    sm = StackedRegressor([('be', base_reg), ('me', meta_reg)])

    assert sm.meta_estimator == meta_reg
    assert sm.meta_estimator_name == 'me'
    assert sm.named_base_estimators == {'be': base_reg}


@pytest.mark.parametrize('n_jobs', [1, 3])
def test_smoke_clf_methods(clf_test_data, n_jobs):
    """Construct, fit, and predict on realistic problem.
    """
    xtrain = clf_test_data['x']
    ytrain = clf_test_data['y']

    rng = np.random.RandomState(17)
    est_list = [('lr', LogisticRegression(C=10**6, random_state=rng,
                                          solver='lbfgs')),
                ('rf', RandomForestClassifier(random_state=rng,
                                              n_estimators=10)),
                ('metalr', LogisticRegression(random_state=rng,
                                              solver='lbfgs'))]
    sm = StackedClassifier(est_list, n_jobs=n_jobs)
    sm.fit(xtrain, ytrain)
    sm.predict(xtrain)
    sm.predict_proba(xtrain)
    sm.predict_log_proba(xtrain)
    sm.decision_function(xtrain)
    sm.score(xtrain, ytrain)
    sm.classes_


@pytest.mark.parametrize('n_jobs', [1, 3])
def test_smoke_multiclass_clf_methods(clf_test_data, n_jobs):
    """Construct, fit, and predict on realistic problem.
    """
    rng = np.random.RandomState(17)
    X, y = make_classification(n_classes=4, n_informative=4, random_state=rng)
    est_list = [('dt', DecisionTreeClassifier(random_state=rng)),
                ('rf', RandomForestClassifier(random_state=rng,
                                              n_estimators=10)),
                ('metarf', RandomForestClassifier(random_state=rng,
                                                  n_estimators=10))]
    sm = StackedClassifier(est_list, n_jobs=n_jobs)
    sm.fit(X, y)
    sm.predict(X)
    sm.predict_proba(X)
    sm.predict_log_proba(X)
    sm.score(X, y)
    sm.classes_


@pytest.mark.parametrize('n_jobs', [1, 3])
def test_smoke_regression_methods(regression_test_data, n_jobs):
    """Construct, fit, and predict on realistic problem.
    """
    xtrain = regression_test_data['x']
    ytrain = regression_test_data['y']

    rng = np.random.RandomState(17)
    est_list = [('lr', LinearRegression()),
                ('rf', RandomForestRegressor(random_state=rng,
                                             n_estimators=10)),
                ('nnls', NonNegativeLinearRegression())]
    sm = StackedRegressor(est_list, n_jobs=n_jobs)
    sm.fit(xtrain, ytrain)
    sm.predict(xtrain)
    sm.score(xtrain, ytrain)

    with pytest.raises(AttributeError):
        sm.predict_proba(xtrain)


@pytest.mark.parametrize('n_jobs', [1, 3])
def test_smoke_multiout_regression_methods(n_jobs):
    """Construct, fit, and predict on realistic problem.
    """
    X, y = make_regression(random_state=7, n_samples=100, n_features=10,
                           n_informative=4, n_targets=2)

    rng = np.random.RandomState(17)
    est_list = [('lr', LinearRegression()),
                ('rf', RandomForestRegressor(random_state=rng,
                                             n_estimators=10)),
                ('metalr', LinearRegression())]
    sm = StackedRegressor(est_list, n_jobs=n_jobs)
    sm.fit(X, y)
    sm.predict(X)
    sm.score(X, y)

    with pytest.raises(AttributeError):
        sm.predict_proba(X)


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_validate_estimators(SM):
    bad_ests1 = [('be', NoFit()), ('me', ClfEst())]
    bad_ests2 = [('be', ClfEst()), ('me', NoFit())]

    # estimators should have a fit method
    for est_list in [bad_ests1, bad_ests2]:
        sm = SM(est_list)
        errmsg = 'does not have fit method'
        with pytest.raises(TypeError) as err:
            sm._validate_estimators()
        assert errmsg in str(err.value)


def test_validate_clf_estimators():
    bad_clfs1 = [('be', Transformer()), ('me', BasicEst())]
    bad_clfs2 = [('be', BasicEst()), ('me', Transformer())]

    errmsg = "does not have `predict_prob`, `decision_function`, or `predict`"
    for est_list in [bad_clfs1, bad_clfs2]:
        sm = StackedClassifier(est_list)
        with pytest.raises(RuntimeError) as runerr:
            sm._validate_estimators()
        assert errmsg in str(runerr.value)


def test_check_clf_methods():
    est = Transformer()
    with pytest.raises(RuntimeError):
        stacking._check_classifier_methods(est, 'bad_est')

    est = BasicEst()
    stacking._check_classifier_methods(est, 'ok_est')


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_validate_estimators_in_fit(SM):
    """This makes sure estimators are validated in the `fit` method itself.
    """
    bad_clfs1 = [('be', ClfEst()), ('me', Transformer())]
    bad_clfs2 = [('be', Transformer()), ('me', ClfEst())]

    # clfs should have either predict_proba, decision_function, or predict
    # method
    if SM == StackedClassifier:
        errmsg = "does not have `predict_prob`, `decision_function`, or `pred"
        for est_list in [bad_clfs1, bad_clfs2]:
            sm = StackedClassifier(est_list)
            with pytest.raises(RuntimeError) as runerr:
                sm.fit([[1]], [1])
            assert errmsg in str(runerr.value)

    bad_ests1 = [('be', NoFit()), ('me', ClfEst())]
    bad_ests2 = [('be', ClfEst()), ('me', NoFit())]

    # estimators should have a fit method
    for est_list in [bad_ests1, bad_ests2]:
        sm = SM(est_list)
        errmsg = 'does not have fit method'
        with pytest.raises(TypeError) as err:
            sm.fit([[1]], [1])
        assert errmsg in str(err.value)


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_validate_names(SM):
    # No double underscores in estimator names
    bad_names1 = [('foo__q', BasicEst()), ('bar', BasicEst())]
    # No duplicate names
    bad_names2 = [('foo', BasicEst()), ('foo', BasicEst())]
    exception_strings = ['Estimator names must not contain __',
                         'Names provided are not unique:']

    for bn, estr in zip([bad_names1, bad_names2], exception_strings):
        sm = SM(bn)
        names, estimators = zip(*sm.estimator_list)

        with pytest.raises(ValueError) as valerr:
            sm._validate_names(names)
        assert estr in str(valerr.value)


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_validate_names_in_fit(SM):
    """This makes sure names are validated in the `fit` method itself.
    """
    # No double underscores in estimator names
    bad_names1 = [('foo__q', BasicEst()), ('bar', BasicEst())]
    # No duplicate names
    bad_names2 = [('foo', BasicEst()), ('foo', BasicEst())]
    exception_strings = ['Estimator names must not contain __',
                         'Names provided are not unique:']

    for bn, estr in zip([bad_names1, bad_names2], exception_strings):
        sm = SM(bn)

        # Bad names should fail on fit
        with pytest.raises(ValueError) as valerr:
            sm.fit([[1]], [1])
        assert estr in str(valerr.value)

        # Ensure that bad parameter setting after construction fails
        sm = SM([('baz', BasicEst())])
        sm.set_params(**{'estimator_list': bn})

        with pytest.raises(ValueError) as valerr:
            sm.fit([[1]], [1])
        assert estr in str(valerr.value)


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_validate_at_least_2_estimators(SM):
    # estimator_list must have at least 2 estimators
    sm = SM([('be', FullClfEst())])

    with pytest.raises(RuntimeError) as runerr:
        sm._validate_estimators()
    assert 'You must have two or more estimators' in str(runerr.value)

    with pytest.raises(RuntimeError) as runerr:
        sm.fit([[1]], [1])
    assert 'You must have two or more estimators' in str(runerr.value)


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_extract_fit_params(SM):
    sm = SM([('be', BasicEst()), ('me', BasicEst())])

    assert (sm._extract_fit_params(**{'be__a': 7, 'me__foo': 17}) ==
            {'be': {'a': 7}, 'me': {'foo': 17}})

    # There is no estimator called 'other', so there will be a KeyError
    with pytest.raises(KeyError):
        sm._extract_fit_params(**{'be__a': 7, 'me__foo': 17,
                                  'other__thing': 3})

    # Fit params should have double underscore like: 'est__param'
    with pytest.raises(ValueError):
        sm._extract_fit_params(**{'be__a': 7, 'me__foo': 17, 'me': 3})


@pytest.mark.parametrize('SM', [StackedClassifier, StackedRegressor])
def test_base_est_fit(SM):
    estlist = [('be1', ClfEst()),
               ('be2', ClfEst()),
               ('meta', ClfEst())]
    sm = SM(estlist)
    fp = {'be1__foo': 3, 'be2__bar': 7, 'meta__baz': 'wut'}
    X = 234
    y = 567
    sm._base_est_fit(X, y, **fp)

    # Check base estimator fits
    for est in sm.estimator_list[:-1]:
        assert est[1].Xfit == 234
        assert est[1].yfit == 567
        assert est[1].classes_ == [0, 1]

    # Check fit parameters properly dispatched
    assert sm.estimator_list[0][1].fit_params == {'foo': 3}
    assert sm.estimator_list[1][1].fit_params == {'bar': 7}

    # Check meta-estimator remains unfit
    with pytest.raises(AttributeError):
        sm.meta_estimator.classes_


def test_base_est_predict_clf():
    estlist = [('be1', FullClfEst()),
               ('be2', ClfEst()),
               ('meta', ClfEst())]
    sm = StackedClassifier(estlist)
    Xmeta = sm._base_est_predict(np.array([[-1], [-2], [-3]]))

    # Note that the StackedClassifier should strip off the last column from the
    # ouptut ClfEst's of predict_proba, since the rows of predict_proba always
    # sum to 1.
    assert_array_equal(Xmeta, np.array([[1, 1, 2],
                                        [2, 4, 5],
                                        [3, 7, 8]]))


def test_base_est_predict_reg():
    estlist = [('be1', BasicEst()),
               ('be2', BasicEst()),
               ('meta', BasicEst())]
    sm = StackedRegressor(estlist)
    Xmeta = sm._base_est_predict(np.array([[-1], [-2], [-3]]))

    assert_array_equal(Xmeta, np.array([[3, 4, 3, 4],
                                        [5, 6, 5, 6],
                                        [7, 8, 7, 8]]))


@mock.patch('civismlext.stacking.clone', lambda x: x)
def test_base_est_fit_predict_clf():
    estlist = [('be1', PassThruClf()),
               ('be2', PassThruDF()),
               ('meta', PassThruClf())]
    sc = StackedClassifier(estlist, cv=2)
    fit_params = {'be1__foo': 'f', 'meta__bar': 'b'}
    x = np.arange(16).reshape((8, 2))
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1])

    xmeta, ymeta, fps = sc._base_est_fit_predict(x, y, **fit_params)
    assert_array_equal(ymeta, y)
    # Note that the StackedClassifier should strip off the last column from the
    # ouptut ClfEst's of predict_proba, since the rows of predict_proba always
    # sum to 1.
    assert_array_equal(xmeta, np.array([[8.,  8.,  9.],
                                        [10., 10., 11.],
                                        [12., 12., 13.],
                                        [14., 14., 15.],
                                        [0.,  0.,  1.],
                                        [2.,  2.,  3.],
                                        [4.,  4.,  5.],
                                        [6.,  6.,  7.]]))
    assert sc.estimator_list[0][1].fit_params == {'foo': 'f'}
    assert sc.estimator_list[1][1].fit_params == {}
    assert fps == {'bar': 'b'}

    # Meta estimator should have never been fit
    with pytest.raises(AttributeError):
        sc.meta_estimator.Xfit
    with pytest.raises(AttributeError):
        sc.meta_estimator.fit_params


@mock.patch('civismlext.stacking.clone', lambda x: x)
def test_base_est_fit_predict_multiout_clf():
    estlist = [('be1', PassThruClf()),
               ('be2', PassThruDF()),
               ('meta', PassThruClf())]
    sc = StackedClassifier(estlist, cv=2)
    fit_params = {'be1__foo': 'f', 'meta__bar': 'b'}
    x = np.arange(12).reshape((6, 2))
    y = np.array([0, 1, 2, 1, 0, 2])

    xmeta, ymeta, fps = sc._base_est_fit_predict(x, y, **fit_params)
    assert_array_equal(ymeta, y)
    assert_array_equal(xmeta, np.array([[6.,  6.,  7.],
                                        [8.,  8.,  9.],
                                        [10., 10., 11.],
                                        [0.,  0.,  1.],
                                        [2.,  2.,  3.],
                                        [4.,  4.,  5.]]))
    assert sc.estimator_list[0][1].fit_params == {'foo': 'f'}
    assert sc.estimator_list[1][1].fit_params == {}

    # Meta estimator should have never been fit
    with pytest.raises(AttributeError):
        sc.meta_estimator.Xfit
    with pytest.raises(AttributeError):
        sc.meta_estimator.fit_params


def test_fit_clf():
    estlist = [('be1', PassThruClf()),
               ('be2', PassThruDF()),
               ('meta', PassThruClf())]
    sc = StackedClassifier(estlist, cv=2)
    fit_params = {'be1__foo': 'f', 'meta__bar': 'b'}
    x = np.arange(16).reshape((8, 2))
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1])

    sc.fit(x, y, **fit_params)
    xmeta = sc.meta_estimator.Xfit
    ymeta = sc.meta_estimator.yfit
    assert_array_equal(ymeta, y)
    assert_array_equal(xmeta, np.array([[8.,  8.,  9.],
                                        [10., 10., 11.],
                                        [12., 12., 13.],
                                        [14., 14., 15.],
                                        [0.,  0.,  1.],
                                        [2.,  2.,  3.],
                                        [4.,  4.,  5.],
                                        [6.,  6.,  7.]]))
    for name, est in sc.estimator_list[:-1]:
        assert_array_equal(est.Xfit, x)
        assert_array_equal(est.yfit, y)
    assert sc.estimator_list[0][1].fit_params == {'foo': 'f'}
    assert sc.estimator_list[1][1].fit_params == {}
    assert sc.meta_estimator.fit_params == {'bar': 'b'}


def test_fit_multiclass_clf():
    estlist = [('be1', PassThruClf()),
               ('be2', PassThruDF()),
               ('meta', PassThruClf())]
    sc = StackedClassifier(estlist, cv=2)
    fit_params = {'be1__foo': 'f', 'meta__bar': 'b'}
    x = np.arange(12).reshape((6, 2))
    y = np.array([0, 1, 2, 1, 0, 2])

    sc.fit(x, y, **fit_params)
    xmeta = sc.meta_estimator.Xfit
    ymeta = sc.meta_estimator.yfit
    assert_array_equal(ymeta, y)
    assert_array_equal(xmeta, np.array([[6.,  6.,  7.],
                                        [8.,  8.,  9.],
                                        [10., 10., 11.],
                                        [0.,  0.,  1.],
                                        [2.,  2.,  3.],
                                        [4.,  4.,  5.]]))
    for name, est in sc.estimator_list[:-1]:
        assert_array_equal(est.Xfit, x)
        assert_array_equal(est.yfit, y)
    assert sc.estimator_list[0][1].fit_params == {'foo': 'f'}
    assert sc.estimator_list[1][1].fit_params == {}
    assert sc.meta_estimator.fit_params == {'bar': 'b'}


def test_fit_multiout_clf():
    estlist = [('be1', PassThruClf()),
               ('be2', PassThruDF()),
               ('meta', PassThruClf())]
    sc = StackedClassifier(estlist)
    X = np.arange(3).reshape((3, 1))
    y = np.array([[0, 1],
                  [1, 0],
                  [0, 1]])
    with pytest.raises(NotImplementedError):
        sc.fit(X, y)


@mock.patch('civismlext.stacking.clone', lambda x: x)
def test_base_est_fit_predict_regression():
    estlist = [('be1', PassThruReg()),
               ('be2', PassThruReg()),
               ('meta', PassThruReg())]
    sr = StackedRegressor(estlist, cv=2)
    fit_params = {'be1__foo': 'f', 'meta__bar': 'b'}
    x = np.arange(12).reshape((6, 2))
    y = np.arange(6)

    xmeta, ymeta, fps = sr._base_est_fit_predict(x, y, **fit_params)
    assert_array_equal(ymeta, y)
    assert_array_equal(xmeta, np.array([[6.,  7.,  6.,  7.],
                                        [8.,  9.,  8.,  9.],
                                        [10., 11., 10., 11.],
                                        [0.,  1.,  0.,  1.],
                                        [2.,  3.,  2.,  3.],
                                        [4.,  5.,  4.,  5.]]))
    assert sr.estimator_list[0][1].fit_params == {'foo': 'f'}
    assert sr.estimator_list[1][1].fit_params == {}

    # Meta estimator should have never been fit
    with pytest.raises(AttributeError):
        sr.meta_estimator.Xfit
    with pytest.raises(AttributeError):
        sr.meta_estimator.fit_params


def test_fit_regression():
    estlist = [('be1', PassThruReg()),
               ('be2', PassThruReg()),
               ('meta', PassThruReg())]
    sr = StackedRegressor(estlist, cv=2)
    fit_params = {'be1__foo': 'f', 'meta__bar': 'b'}
    x = np.arange(12).reshape((6, 2))
    y = np.arange(6)

    sr.fit(x, y, **fit_params)

    xmeta = sr.meta_estimator.Xfit
    ymeta = sr.meta_estimator.yfit
    assert_array_equal(ymeta, y)
    assert_array_equal(xmeta, np.array([[6.,  7.,  6.,  7.],
                                        [8.,  9.,  8.,  9.],
                                        [10., 11., 10., 11.],
                                        [0.,  1.,  0.,  1.],
                                        [2.,  3.,  2.,  3.],
                                        [4.,  5.,  4.,  5.]]))
    assert sr.estimator_list[0][1].fit_params == {'foo': 'f'}
    assert sr.estimator_list[1][1].fit_params == {}
    for name, est in sr.estimator_list[:-1]:
        assert_array_equal(est.Xfit, x)
        assert_array_equal(est.yfit, y)
    assert sr.estimator_list[0][1].fit_params == {'foo': 'f'}
    assert sr.estimator_list[1][1].fit_params == {}
    assert sr.meta_estimator.fit_params == {'bar': 'b'}


def test_cv_shuffle_indices():
    """Make sure xmeta and ymeta retain the correct order, even when the CV
    generator is shuffling. This is checking for the bug reported in issue #16.
    """
    estlist = [('be1', PassThruReg()),
               ('be2', PassThruReg()),
               ('meta', PassThruReg())]
    sr = StackedRegressor(estlist, cv=KFold(n_splits=2, shuffle=True))
    x = np.arange(6)
    y = np.arange(6)

    # Suppose the train indices of a 2-fold CV are:
    #  [a, b, c]  and  [d, e, f]
    # Then the test indices are:
    #  [d, e, f]  and  [a, b, c]
    # Since xmeta is just a pass-through of x[train] (horizontally stacked
    # twice, due to the two base estimators) and ymeta is a pass-through of
    # y[test], and since x = y, we should expect that:
    #  xmeta[inds, 0] == ymeta
    #
    inds = np.array([3, 4, 5, 0, 1, 2])

    xmeta, ymeta, _ = sr._base_est_fit_predict(x, y)
    np.testing.assert_equal(xmeta[inds, 0], ymeta)


@mock.patch('civismlext.stacking.clone', lambda x: x)
def test_base_est_fit_predict_multiout_regression():
    estlist = [('be1', PassThruReg()),
               ('be2', PassThruReg()),
               ('meta', PassThruReg())]
    sr = StackedRegressor(estlist, cv=2)
    fit_params = {'be1__foo': 'f', 'meta__bar': 'b'}
    x = np.arange(12).reshape((6, 2))
    y = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8],
                  [0.9, 1.0], [1.1, 1.2]])

    xmeta, ymeta, fps = sr._base_est_fit_predict(x, y, **fit_params)
    assert_array_equal(ymeta, y)
    assert_array_equal(xmeta, np.array([[6.,  7.,  6.,  7.],
                                        [8.,  9.,  8.,  9.],
                                        [10., 11., 10., 11.],
                                        [0.,  1.,  0.,  1.],
                                        [2.,  3.,  2.,  3.],
                                        [4.,  5.,  4.,  5.]]))
    assert sr.estimator_list[0][1].fit_params == {'foo': 'f'}
    assert sr.estimator_list[1][1].fit_params == {}

    # Meta estimator should have never been fit
    with pytest.raises(AttributeError):
        sr.meta_estimator.Xfit
    with pytest.raises(AttributeError):
        sr.meta_estimator.fit_params


def test_fit_multiout_regression():
    estlist = [('be1', PassThruReg()),
               ('be2', PassThruReg()),
               ('meta', PassThruReg())]
    sr = StackedRegressor(estlist, cv=2)
    fit_params = {'be1__foo': 'f', 'meta__bar': 'b'}
    x = np.arange(12).reshape((6, 2))
    y = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8],
                  [0.9, 1.0], [1.1, 1.2]])

    sr.fit(x, y, **fit_params)

    xmeta = sr.meta_estimator.Xfit
    ymeta = sr.meta_estimator.yfit
    assert_array_equal(ymeta, y)
    assert_array_equal(xmeta, np.array([[6.,  7.,  6.,  7.],
                                        [8.,  9.,  8.,  9.],
                                        [10., 11., 10., 11.],
                                        [0.,  1.,  0.,  1.],
                                        [2.,  3.,  2.,  3.],
                                        [4.,  5.,  4.,  5.]]))
    assert sr.estimator_list[0][1].fit_params == {'foo': 'f'}
    assert sr.estimator_list[1][1].fit_params == {}
    for name, est in sr.estimator_list[:-1]:
        assert_array_equal(est.Xfit, x)
        assert_array_equal(est.yfit, y)
    assert sr.meta_estimator.fit_params == {'bar': 'b'}


def test_fit_pred_simple_regression():
    X = np.arange(6).reshape(-1, 1)
    y = np.arange(1, 7)

    slr = StackedRegressor([('lr1', LinearRegression()),
                            ('lr2', LinearRegression()),
                            ('metalr', NonNegativeLinearRegression())])

    slr.fit(X, y)
    # check that base coefs and ints = 1
    for est in slr.estimator_list[:-1]:
        assert_almost_equal(est[1].coef_, np.array([1]))
        assert_almost_equal(est[1].intercept_, 1)
    # check that sum of meta coefs = 1
    assert_almost_equal(slr.meta_estimator.coef_.sum(), 1)
    assert_almost_equal(slr.meta_estimator.intercept_, 0)


def test_fit_pred_simple_clf():
    X = np.arange(6).reshape(-1, 1)
    X = np.vstack((X, X))
    y = (X > 2).astype(int).ravel()

    sc = StackedClassifier([('dt1', DecisionTreeClassifier()),
                            ('dt2', DecisionTreeClassifier()),
                            ('metadt', DecisionTreeClassifier())])

    sc.fit(X, y)
    # All events below 2.5 should have prob=0; above prob=1
    for basetree in sc.estimator_list[:-1]:
        assert_almost_equal(basetree[1].predict_proba(np.array([[2.4]])),
                            np.array([[1, 0]]))
        assert_almost_equal(basetree[1].predict_proba(np.array([[2.6]])),
                            np.array([[0, 1]]))
    assert_almost_equal(sc.predict_proba(np.array([[2.4]])),
                        np.array([[1, 0]]))
    assert_almost_equal(sc.predict_proba(np.array([[2.6]])),
                        np.array([[0, 1]]))


def test_fit_params_regression(regression_test_data):
    xtrain = regression_test_data['x']
    ytrain = regression_test_data['y']
    sample_weights = [1./len(ytrain)] * len(ytrain)
    fit_params = {'rf__sample_weight': sample_weights}
    sr = StackedRegressor([('rf', RandomForestRegressor(random_state=7,
                                                        n_estimators=10)),
                           ('rf2', RandomForestRegressor(n_estimators=10)),
                           ('meta', NonNegativeLinearRegression())])
    Xmeta, ymeta, _ = sr._base_est_fit_predict(
        xtrain, ytrain, **fit_params)
    assert Xmeta.shape == xtrain.shape


def test_fit_params_clf(clf_test_data):
    xtrain = clf_test_data['x']
    ytrain = clf_test_data['y']
    sample_weights = [1./len(ytrain)] * len(ytrain)
    fit_params = {'rf__sample_weight': sample_weights}
    sr = StackedClassifier([('rf', RandomForestClassifier(random_state=7,
                                                          n_estimators=10)),
                           ('lr', LogisticRegression(solver='lbfgs')),
                           ('meta', LogisticRegression(solver='lbfgs'))])
    Xmeta, ymeta, _ = sr._base_est_fit_predict(
        xtrain, ytrain, **fit_params)
    assert Xmeta.shape == xtrain.shape


def fit_predict_measure_reg(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    return mean_squared_error(ytest, ypred)


@pytest.mark.parametrize('n_jobs', [1, 3])
def test_integration_regression(regression_test_data, n_jobs):
    """Construct, fit, and predict on realistic problem. Compare goodness of
    fit of stacked model vs. individual base estimators.
    """
    xtrain = regression_test_data['x']
    ytrain = regression_test_data['y']
    xtest = regression_test_data['xtest']
    ytest = regression_test_data['ytest']

    sr = StackedRegressor([('rf', RandomForestRegressor(random_state=7,
                                                        n_estimators=10)),
                           ('lr', LinearRegression()),
                           ('metalr', NonNegativeLinearRegression())],
                          n_jobs=n_jobs)
    rf = RandomForestRegressor(random_state=7, n_estimators=10)
    lr = LinearRegression()
    sr_mse = fit_predict_measure_reg(sr, xtrain, ytrain, xtest, ytest)
    rf_mse = fit_predict_measure_reg(rf, xtrain, ytrain, xtest, ytest)
    lr_mse = fit_predict_measure_reg(lr, xtrain, ytrain, xtest, ytest)

    # Stacked regressor should perform better than its base estimators on this
    # data.
    assert sr_mse < rf_mse
    assert sr_mse < lr_mse
    assert sr_mse < 1.5    # Sanity check


def fit_predict_measure_clf(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ypred = model.predict_proba(xtest)
    fpr, tpr, _ = roc_curve(ytest, ypred[:, 1])
    return auc(fpr, tpr)


@pytest.mark.parametrize('n_jobs', [1, 3])
def test_integration_clf(clf_test_data, n_jobs):
    """Construct, fit, and predict on realistic problem. Compare goodness of
    fit of stacked model vs. individual base estimators.
    """
    xtrain = clf_test_data['x']
    ytrain = clf_test_data['y']
    xtest = clf_test_data['xtest']
    ytest = clf_test_data['ytest']

    sc = StackedClassifier([('rf', RandomForestClassifier(n_estimators=10)),
                            ('lr', LogisticRegression(solver='lbfgs')),
                            ('metalr', LogisticRegression(solver='lbfgs'))],
                           n_jobs=n_jobs)
    sc.set_params(rf__random_state=7, rf__n_estimators=20,
                  lr__random_state=8, metalr__random_state=9,
                  lr__C=10**7, metalr__C=10**7)
    lr = LogisticRegression(C=10**7, random_state=8, solver='lbfgs')
    rf = RandomForestClassifier(n_estimators=20, random_state=7)

    sc_auc = fit_predict_measure_clf(sc, xtrain, ytrain, xtest, ytest)
    lr_auc = fit_predict_measure_clf(lr, xtrain, ytrain, xtest, ytest)
    rf_auc = fit_predict_measure_clf(rf, xtrain, ytrain, xtest, ytest)

    # Sanity check the AUCs of the base estimators
    assert lr_auc > 0.6
    assert rf_auc > 0.6
    # Stacked classifier should perform better than its base estimators on this
    # data.
    assert sc_auc > lr_auc
    assert sc_auc > rf_auc
