from __future__ import print_function
from __future__ import division

import pytest

from scipy.stats import expon, randint, rankdata
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils import check_random_state
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from ..hyperband import HyperbandSearchCV, hyperband_num_per_run


class DummyCVedEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, a=1, b=100):
        self.a = a
        self.b = b

    def predict(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        self._rng = check_random_state(self.a)
        return self

    def score(self, *args, **kwargs):
        return self._rng.uniform() + self.b


@pytest.mark.parametrize('min_iter', [None, 1, 3])
def test_smoke_hyperband(min_iter):
    seed = 10
    n_splits = 4
    eta = 3
    max_iter = 27
    params = dict(a=randint(low=1, high=100))
    cmax_param = {'b': max_iter}
    if min_iter is not None:
        cmin_param = {'b': min_iter}
    else:
        cmin_param = None
    hyperband_search = HyperbandSearchCV(
        DummyCVedEstimator(),
        cost_parameter_max=cmax_param,
        cost_parameter_min=cmin_param,
        cv=n_splits,
        iid=False,
        return_train_score=False,
        eta=eta,
        param_distributions=params,
        random_state=seed)

    hyperband_search.fit(np.random.normal(size=(100, 3)),
                         np.random.choice(2, size=100, replace=True))

    rng = np.random.RandomState(seed=seed)
    ri = randint(low=1, high=100)

    # b_vals is the geometric sequence of values of b hyperband will test
    if min_iter is None or min_iter == 1:
        b_vals = np.array(
            [1] * 27 + [3] * 9 + [9] * 3 + [27] +
            [3] * 12 + [9] * 4 + [27] +
            [9] * 6 + [27] * 2 +
            [27] * 4)
        a_itrs = [[27, 9, 3, 1], [12, 4, 1], [6, 2], [4]]
    else:
        b_vals = np.array(
            [3] * 9 + [9] * 3 + [27] +
            [9] * 5 + [27] +
            [27] * 3)
        a_itrs = [[9, 3, 1], [5, 1], [3]]

    # now draw the a_vals
    # this will be a list of list of lists of the
    # indices for each score
    inds_to_reorder = []
    loc = 0  # counter used to track the next index
    a_vals = []
    for bstart, nks in enumerate(a_itrs):
        a_vals_orig = ri.rvs(random_state=rng, size=nks[0]).tolist()
        # this first element is a duplicate that is removed below
        _inds_to_reorder = [list(np.arange(loc, loc+nks[0]))]
        for i, nk in enumerate(nks):
            scores = np.array(
                [np.random.RandomState(seed=s).uniform()
                 for s in a_vals_orig]) + np.power(3, i + bstart)
            sinds = np.argsort(scores)[::-1]  # bigger is better
            msk = np.zeros_like(a_vals_orig)
            msk[sinds[0:nk]] = 1
            msk = msk.astype(bool)
            a_vals_orig = [a for i, a in enumerate(a_vals_orig) if msk[i]]
            a_vals += a_vals_orig

            # now add the extra inds
            _inds_to_reorder.append(list(np.arange(loc, loc+nk)))
            loc += nk

        # add the list of lists to the total
        inds_to_reorder.append(_inds_to_reorder[1:])

    a_vals = np.array(a_vals)
    mn_scores = np.array(
        [np.random.RandomState(seed=a).uniform() for a in a_vals]) + b_vals

    # check that we get the same set of scores back
    # a useful test and helps for debugging
    assert (
        set(mn_scores) ==
        set(hyperband_search.cv_results_['mean_test_score'])), (
        "The set of returned scores is wrong!")

    # finally we have to swap the order of the inner and outer loops above
    # to match our implementation of hyperband
    # we have kept the inds corresponding to each iteration, so we just need
    # to add up the lists in the right order
    final_inds = []
    max_inner = max([len(a) for a in inds_to_reorder])
    for rnd in range(max_inner):
        for inds in inds_to_reorder:
            if rnd < len(inds):
                final_inds += inds[rnd]
    # and reorder the outputs
    mn_scores = mn_scores[final_inds]
    a_vals = a_vals[final_inds]
    b_vals = b_vals[final_inds]
    best_index = np.argmax(mn_scores)

    # now make sure it got the right values
    assert hyperband_search.best_index_ == best_index, "Best index is wrong!"
    assert hyperband_search.best_score_ == mn_scores[best_index], (
        "Best score is wrong!")
    assert (hyperband_search.best_params_ ==
            {'a': a_vals[best_index], 'b': b_vals[best_index]}), (
        "Best parameters are wrong!")


def check_cv_results_array_types(cv_results, param_keys, score_keys):
    # Check if the search `cv_results`'s array are of correct types
    assert (all(isinstance(cv_results[param], np.ma.MaskedArray)
                for param in param_keys))
    assert (all(cv_results[key].dtype == object for key in param_keys))
    assert (any(isinstance(cv_results[key], np.ma.MaskedArray)
                for key in score_keys)) is False
    assert (all(cv_results[key].dtype == np.float64
                for key in score_keys if not key.startswith('rank')))
    assert (cv_results['rank_test_score'].dtype == np.int32)


def check_cv_results_keys(cv_results, param_keys, score_keys, n_cand):
    # Test the search.cv_results_ contains all the required results
    assert_array_equal(sorted(cv_results.keys()),
                       sorted(param_keys + score_keys + ('params',)))
    assert (all(cv_results[key].shape == (n_cand,)
                for key in param_keys + score_keys))


@pytest.mark.parametrize('min_iter', [None, 1, 100])
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_hyperband_search_cv_results(min_iter):
    # Make a dataset with a lot of noise to get various kind of prediction
    # errors across CV folds and parameter settings
    X, y = make_classification(n_samples=200, n_features=100, n_informative=3,
                               random_state=0)

    # scipy.stats dists now supports `seed` but sklearn
    # still supports scipy 0.12 which doesn't support the seed.
    # Hence the assertions in the test for hyperband alone should not
    # depend on randomization.
    n_splits = 3
    eta = 3
    max_iter = 1000
    n_cand = hyperband_num_per_run(
        eta,
        max_iter,
        min_iter if min_iter is not None else 1)
    params = dict(C=expon(scale=10), gamma=expon(scale=0.1))
    cmax_param = {'max_iter': max_iter}
    if min_iter is not None:
        cmin_param = {'min_iter': min_iter}
    else:
        cmin_param = None
    hyperband_search = HyperbandSearchCV(
        SVC(),
        cost_parameter_max=cmax_param,
        cost_parameter_min=cmin_param,
        cv=n_splits,
        iid=False,
        eta=eta,
        param_distributions=params)
    hyperband_search.fit(X, y)
    hyperband_search_iid = HyperbandSearchCV(
        SVC(),
        cost_parameter_max=cmax_param,
        cost_parameter_min=cmin_param,
        cv=n_splits,
        iid=True,
        eta=eta,
        param_distributions=params)
    hyperband_search_iid.fit(X, y)

    param_keys = ('param_C', 'param_gamma', 'param_max_iter')
    score_keys = ('mean_test_score', 'mean_train_score',
                  'rank_test_score',
                  'split0_test_score', 'split1_test_score',
                  'split2_test_score',
                  'split0_train_score', 'split1_train_score',
                  'split2_train_score',
                  'std_test_score', 'std_train_score',
                  'mean_fit_time', 'std_fit_time',
                  'mean_score_time', 'std_score_time')

    for search, iid in zip(
            (hyperband_search, hyperband_search_iid), (False, True)):
        assert_equal(iid, search.iid)
        cv_results = search.cv_results_
        # Check results structure
        check_cv_results_array_types(cv_results, param_keys, score_keys)
        check_cv_results_keys(cv_results, param_keys, score_keys, n_cand)
        # For random_search, all the param array vals should be unmasked
        assert (any(cv_results['param_C'].mask) or
                any(cv_results['param_gamma'].mask)) is False


@pytest.mark.parametrize('rank', [False, True])
@pytest.mark.parametrize('splits', [False, True])
@pytest.mark.parametrize('weights', [None, np.array([0.1, 0.4, 0.5])])
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_store_results_method(weights, splits, rank):
    rng = np.random.RandomState(seed=10)

    n_candidates = 10
    n_splits = 3
    hyperband_search = HyperbandSearchCV(
        SVC(),
        cost_parameter_max={},
        param_distributions={})

    array = rng.uniform(size=n_candidates * n_splits)
    mns = np.average(array.reshape(10, 3), axis=1, weights=weights)
    stds = np.sqrt(np.average(
        (array.reshape(10, 3) - mns.reshape(-1, 1)) ** 2,
        axis=1,
        weights=weights))
    ranks = np.array(rankdata(-mns, method='min'), dtype=np.int32)
    key_name = 'abc'
    results = {}

    results = hyperband_search._store_results(
        results,
        n_splits,
        n_candidates,
        key_name,
        array,
        weights=weights,
        splits=splits,
        rank=rank)

    if splits:
        for i in range(n_splits):
            assert np.allclose(
                results['split%d_abc' % i],
                array.reshape(10, 3)[:, i]), (
                "Array values for split %d are not right!" % i)

    assert np.allclose(results['mean_abc'], mns), "Means are not correct!"
    assert np.allclose(results['std_abc'], stds), "Stds are not correct!"

    if rank:
        assert np.array_equal(results['rank_abc'], ranks), (
            "Ranks are not correct!")


@pytest.fixture(params=[True, False])
def out(request):
    if request.param:
        # this bool indicates if the train scores are returned as first
        # element of the tuples
        return [(0.1, 0.4, 10, 1, 2, {'a': 1}),
                (0.2, 0.5, 20, 3, 4, {'a': 1}),
                (0.3, 0.6, 11, 5, 6, {'a': 1}),
                (0.7, 1.0, 10, 7, 8, {'a': 2, 'b': 6}),
                (0.8, 1.1, 20, 9, 10, {'a': 2, 'b': 6}),
                (0.9, 1.2, 11, 11, 12, {'a': 2, 'b': 6})]
    else:
        return [(0.4, 10, 1, 2, {'a': 1}),
                (0.5, 20, 3, 4, {'a': 1}),
                (0.6, 11, 5, 6, {'a': 1}),
                (1.0, 10, 7, 8, {'a': 2, 'b': 6}),
                (1.1, 20, 9, 10, {'a': 2, 'b': 6}),
                (1.2, 11, 11, 12, {'a': 2, 'b': 6})]


@pytest.mark.parametrize('iid', [False, True])
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
def test_process_outputs_method(out, iid):
    return_train_score = len(out[0]) == 6  # train scores are returned?
    hyperband_search = HyperbandSearchCV(
        SVC(),
        cost_parameter_max={},
        param_distributions={},
        return_train_score=return_train_score,
        iid=iid)

    if iid:
        wts = np.array([10, 20, 11]) / 41
    else:
        wts = None

    score_mns = [
        np.average([0.4, 0.5, 0.6], weights=wts),
        np.average([1.0, 1.1, 1.2], weights=wts)]

    results, best_index = hyperband_search._process_outputs(
        out, n_splits=3)

    assert best_index == 1, (
        "The best candidate index was not computed correctly!")

    assert results['params'] == ({'a': 1}, {'a': 2, 'b': 6})
    assert np.array_equal(
        results['param_a'],
        np.ma.array(np.array([1, 2]), mask=[0, 0], dtype=object)), (
        "Param a results are not correct!")
    # use np.all here since masked array comparison will ignore the masked
    # element - np.array_equal does not for some reason...
    assert np.all(results['param_b'] ==
                  np.ma.array([None, 6], mask=[True, False], dtype=object)), (
            "Param b results are not correct!")

    assert np.allclose(results['split0_test_score'], [0.4, 1.0]), (
        "Split 0 test score is wrong!")
    assert np.allclose(results['split1_test_score'], [0.5, 1.1]), (
        "Split 1 test score is wrong!")
    assert np.allclose(results['split2_test_score'], [0.6, 1.2]), (
        "Split 2 test score is wrong!")
    assert np.allclose(results['mean_test_score'], score_mns), (
        "Mean test score is not correct!")
    assert np.allclose(
        results['std_test_score'],
        [np.sqrt(np.average(
            (np.array([0.4, 0.5, 0.6]) - score_mns[0]) ** 2, weights=wts)),
         np.sqrt(np.average(
            (np.array([1.0, 1.1, 1.2]) - score_mns[1]) ** 2, weights=wts))]), (
        "Std of test score is not correct!")
    assert np.allclose(results['rank_test_score'], [2, 1]), (
        "Test score ranks are wrong!")

    if return_train_score:
        assert np.allclose(results['split0_train_score'], [0.1, 0.7]), (
            "Split 0 train score is wrong!")
        assert np.allclose(results['split1_train_score'], [0.2, 0.8]), (
            "Split 1 train score is wrong!")
        assert np.allclose(results['split2_train_score'], [0.3, 0.9]), (
            "Split 2 train score is wrong!")
        assert np.allclose(results['mean_train_score'], [0.2, 0.8]), (
            "Mean train score is not correct!")
        assert np.allclose(
            results['std_train_score'],
            [np.sqrt(np.average(
                (np.array([0.1, 0.2, 0.3]) - 0.2) ** 2)),
             np.sqrt(np.average(
                (np.array([0.7, 0.8, 0.9]) - 0.8) ** 2))]), (
            "Std of train score is not correct!")

    assert np.allclose(
        results['mean_score_time'],
        [np.mean([2, 4, 6]), np.mean([8, 10, 12])]), (
        "Mean score time is not correct!")

    assert np.allclose(
        results['std_score_time'], [np.std([2, 4, 6]), np.std([8, 10, 12])]), (
        "Std of score time is not correct!")

    assert np.allclose(
        results['mean_fit_time'], [np.mean([1, 3, 5]), np.mean([7, 9, 11])]), (
        "Mean fit time is not correct!")

    assert np.allclose(
        results['std_fit_time'], [np.std([1, 3, 5]), np.std([7, 9, 11])]), (
        "Std of fit time is not correct!")
