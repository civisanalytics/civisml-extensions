from __future__ import print_function
from __future__ import division

import copy
import itertools
import logging

import numpy as np

from numpy.ma import MaskedArray
from joblib import Parallel, delayed
from scipy.stats import rankdata

from sklearn.model_selection import ParameterSampler
from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import is_classifier, clone
from sklearn.model_selection import check_cv

# I don't want this, but fine.
from sklearn.model_selection._validation import _fit_and_score

from sklearn.utils.validation import indexable
from sklearn.metrics.scorer import check_scoring
from sklearn.utils import check_random_state

log = logging.getLogger(__name__)


def hyperband_num_per_run(eta, R, Rmin):
    num = 0
    smax = int(np.floor(np.log(R / Rmin) / np.log(eta)))
    B = (smax + 1.0) * R
    for s in range(smax, -1, -1):
        n = int(np.ceil(B / R * np.power(eta, s) / (s + 1.0)))
        T = [0] * n
        for i in range(0, s + 1):
            n_i = int(np.floor(n / np.power(eta, i)))
            num_to_keep = int(np.floor(n_i / eta))
            num += len(T)
            T = T[0:num_to_keep]

    return num


class HyperbandSearchCV(BaseSearchCV):
    """Hyperband search on hyper parameters.

    If all parameters are presented as a list,
    sampling without replacement is performed. If at least one parameter
    is given as a distribution, sampling with replacement is used.

    It is highly recommended to use continuous distributions for continuous
    parameters.

    Parameters
    ----------
    estimator : estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.
    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
    cost_parameter_max : dict
        Dictionary with the cost parameter name as a key and the maximum
        cost as the value. The cost parameter is the maximum cost per randomly
        selected hyperparameter configuration. Typicaly this parameter would
        be the number of iterations for SGD-like methods or the number of
        trees in an ensemble.
    cost_parameter_min : dict, optional
        Dictionary with the cost parameter name as a key and the minimum
        cost as the value. Defaults to 1 if not given. This parameter is the
        minimum of the cost parameter. This option can be used to force
        hyperband to use, say, at least 100 trees in an ensemble method in
        order to limit the hyperband to reasonable choices in hyperparameter
        space.
    eta : int, default=3
        One over the fraction of configurations that are discarded in each
        round of parameter searching.
    scoring : string, callable or None, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If ``None``, the ``score`` method of the estimator is used.
    n_jobs : int, default=1
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
    iid : boolean, default=True
        If True, the data is assumed to be identically distributed across
        the folds, and the loss minimized is the total loss per sample,
        and not the mean loss across the folds.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    refit : boolean, default=True
        Refit the best estimator with the entire dataset.
        If "False", it is impossible to make predictions using
        this HyperbandSearchCV instance after fitting.
    verbose : integer
        Controls the verbosity: the higher, the more messages.
    random_state : int or RandomState
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
    error_score : 'raise' (default) or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.
    return_train_score : boolean, default=True
        If ``'False'``, the ``cv_results_`` attribute will not include training
        scores.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
        For instance the below given table
        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |        0.8        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |        0.9        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |        0.7        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        will be represented by a ``cv_results_`` dict of::
            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.8, 0.9, 0.7],
            'split1_test_score'  : [0.82, 0.5, 0.7],
            'mean_test_score'    : [0.81, 0.7, 0.7],
            'std_test_score'     : [0.02, 0.2, 0.],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.8, 0.9, 0.7],
            'split1_train_score' : [0.82, 0.5, 0.7],
            'mean_train_score'   : [0.81, 0.7, 0.7],
            'std_train_score'    : [0.03, 0.03, 0.04],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
            'params' : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }
        NOTE that the key ``'params'`` is used to store a list of parameter
        settings dict for all the parameter candidates.
        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if refit=False.
    best_score_ : float
        Score of best_estimator on the left out data.
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.
        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).
    scorer_ : function
        Scorer function used on the held out data to choose the best
        parameters for the model.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is
    `2 * n_jobs`.

    See the original paper here https://arxiv.org/abs/1603.06560. The variable
    names in the code reflect the chosen names in the paper for clarity.

    Examples
    --------
    >>> rng = np.random.RandomState(seed=seed)
    >>> from scipy.stats import uniform
    >>> cv = KFold(
    >>>     n_splits=4,
    >>>     shuffle=True,
    >>>     random_state=rng.randint(1, 100000))
    >>> rf = RandomForestClassifier()
    >>> cvest = HyperbandSearchCV(
        rf,
        param_distributions={'criterion': ['gini', 'entropy'],
                             'max_depth': [None, 1, 4, 8, 16, 32],
                             'clf__max_features': uniform()},
        cost_parameter_max={'n_estimators': 1000},
        cost_parameter_min={'n_estimators': 100},
        eta=3,
        cv=cv,
        random_state=rng.randint(1, 100000),
        scoring='roc_auc')
    >>> cvest.fit(X, y)

    See Also
    --------
    :class:`RandomizedSearchCV`:
        Searches randomly over a predefined parameter space.
    :class:`ParameterSampler`:
        A generator over parameter settins, constructed from
        param_distributions.
    """

    def __init__(self, estimator, param_distributions, cost_parameter_max,
                 cost_parameter_min=None, eta=3,
                 scoring=None, n_jobs=1, iid=True, refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score='raise',
                 return_train_score=True):
        self.param_distributions = param_distributions
        self.cost_parameter_max = cost_parameter_max
        self.cost_parameter_min = cost_parameter_min
        self.eta = eta
        self.random_state = random_state
        super(HyperbandSearchCV, self).__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit on the estimator with randomly drawn parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        self._random_state = check_random_state(self.random_state)
        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        R = list(self.cost_parameter_max.values())[0]

        if self.cost_parameter_min is None:
            Rmin = 1
        else:
            Rmin = list(self.cost_parameter_min.values())[0]

        n_candidates = hyperband_num_per_run(self.eta, R, Rmin)
        log.debug("Fitting %d folds for each of %d candidates, totalling "
                  "%d fits.", n_splits, n_candidates, n_candidates * n_splits)
        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        cv_iter = list(cv.split(X, y, groups))

        out = []
        smax = int(np.floor(np.log(R / Rmin) / np.log(self.eta)))
        B = (smax + 1.0) * R

        # This code is hyperband, but I have swapped the order of the
        # inner and outer loops to expose more parallelism. Fun.
        Ts = []
        ns = []
        rs = []
        for s in range(smax, -1, -1):
            ns.append(int(np.ceil(B / R * np.power(self.eta, s) / (s + 1.0))))
            rs.append(int(R / np.power(self.eta, s)))
            Ts.append(list(ParameterSampler(
                self.param_distributions,
                ns[-1],
                random_state=self._random_state)))
        nums = copy.copy(ns)
        # these are the offsets to the hyperparameter configurations for
        # each value of s in the loop above
        # they get updated as the loop over the different rounds get run
        # below
        offsets = [0] + list(
            np.cumsum(np.array(nums) * n_splits).astype(int))[:-1]

        # iterate the maximum number of times for each resource budget
        # configuration.
        # If we should skip an interation, T will be an empty list
        for rnd in range(0, smax + 1):
            # set the costs for this round
            r_rnd = []
            for ind, s in enumerate(range(smax, -1, -1)):
                _r = int(rs[ind] * np.power(self.eta, rnd))
                r_rnd += [_r] * nums[ind]

            # run the jobs
            _jobs = []
            for parameters, _r in zip(
                    itertools.chain.from_iterable(Ts), r_rnd):
                _parameters = copy.deepcopy(parameters)
                _parameters.update(
                    {list(self.cost_parameter_max.keys())[0]: _r})
                for train, test in cv_iter:
                    _jobs.append(delayed(_fit_and_score)(
                        clone(base_estimator), X, y, self.scorer_,
                        train, test, self.verbose, _parameters,
                        fit_params=fit_params,
                        return_train_score=self.return_train_score,
                        return_n_test_samples=True,
                        return_times=True, return_parameters=True,
                        error_score=self.error_score))
            _out = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=pre_dispatch)(_jobs)
            out += _out

            # now post-process
            new_Ts = []
            new_nums = []
            for ind, s in enumerate(range(smax, -1, -1)):
                n_i = int(np.floor(ns[ind] / np.power(self.eta, rnd)))
                num_to_keep = int(np.floor(n_i / self.eta))
                # keep for next round only if num_to_keep > 0 AND
                # the round after this round will be executed
                # in otherwords, you only need to cut the configurations
                # down by eta if you are going to test them in the next
                # round
                if num_to_keep > 0 and rnd < s:
                    _out_s = _out[
                        offsets[ind]:(offsets[ind] + nums[ind] * n_splits)]
                    results, _ = self._process_outputs(_out_s, n_splits)
                    sind = np.argsort(results["rank_test_score"])
                    msk = np.zeros(len(results['rank_test_score']))
                    msk[sind[0:num_to_keep]] = 1
                    msk = msk.astype(bool)
                    new_Ts.append(
                        [p for k, p in enumerate(results['params']) if msk[k]])
                    new_nums.append(num_to_keep)
                else:
                    new_Ts.append([])
                    new_nums.append(0)

            Ts = new_Ts
            nums = new_nums
            offsets = [0] + list(
                np.cumsum(np.array(nums) * n_splits).astype(int))[:-1]

        results, best_index = self._process_outputs(out, n_splits)
        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits
        self.multimetric_ = False
        if not hasattr(self, 'best_score_'):
            self.best_score_ = results['mean_test_score'][best_index]
        if not hasattr(self, 'best_params_'):
            self.best_params_ = results['params'][best_index]

        if self.refit:
            best_estimator = clone(self.estimator).set_params(
                **self.cv_results_['params'][self.best_index_])

            if y is not None:
                best_estimator.fit(X, y, **fit_params)
            else:
                best_estimator.fit(X, **fit_params)

            self.best_estimator_ = best_estimator

        return self

    def _store_results(
            self,
            results,
            n_splits,
            n_candidates,
            key_name,
            array,
            weights=None,
            splits=False,
            rank=False):
        """A small helper to store the scores/times to the cv_results_"""
        array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                          n_splits)
        if splits:
            for split_i in range(n_splits):
                results["split%d_%s"
                        % (split_i, key_name)] = array[:, split_i]

        array_means = np.average(array, axis=1, weights=weights)
        results['mean_%s' % key_name] = array_means
        # Weighted std is not directly available in numpy
        array_stds = np.sqrt(np.average((array -
                                         array_means[:, np.newaxis]) ** 2,
                                        axis=1, weights=weights))
        results['std_%s' % key_name] = array_stds

        if rank:
            results["rank_%s" % key_name] = np.asarray(
                rankdata(-array_means, method='min'), dtype=np.int32)

        return results

    def _process_outputs(self, out, n_splits):
        """return results dict and best dict for given outputs"""

        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            (train_scores, test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)
        else:
            (test_scores, test_sample_counts,
             fit_time, score_time, parameters) = zip(*out)

        candidate_params = parameters[::n_splits]
        n_candidates = len(candidate_params)

        results = dict()

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)

        results = self._store_results(
            results, n_splits, n_candidates, 'test_score',
            test_scores, splits=True, rank=True,
            weights=test_sample_counts if self.iid else None)
        if self.return_train_score:
            results = self._store_results(
                results, n_splits, n_candidates,
                'train_score', train_scores, splits=True)
        results = self._store_results(
            results, n_splits, n_candidates, 'fit_time', fit_time)
        results = self._store_results(
            results, n_splits, n_candidates, 'score_time', score_time)

        best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]

        # Use one np.ma.MaskedArray and mask places where param not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_vals = {}
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index

                param = "param_" + name
                if param not in param_vals:

                    # Map candidates-to-values. Defaults to a masked np.empty
                    param_vals[param] = MaskedArray(np.empty(n_candidates,),
                                                    mask=True,
                                                    dtype=object)

                param_vals[param][cand_idx] = value

        results.update(param_vals)

        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        return results, best_index
