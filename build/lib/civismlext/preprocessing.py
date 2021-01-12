from __future__ import print_function
from __future__ import division

import logging
import uuid
import warnings
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

log = logging.getLogger(__name__)


class DataFrameETL(BaseEstimator, TransformerMixin):
    """Performs ETL on a dataframe, using the following steps:
    - dropping specified columns
    - performing categorical expansion on a specified set of columns
    - creating dummy columns for missing data in a specified set of columns
    - converting the dataframe to a numpy array or dataframe of float32s.

    Parameters
    ----------
    cols_to_drop : list[str], optional (default: None)
        List of columns to drop from the dataframe.
    cols_to_expand: {"auto", None, list[str]}, (default: 'auto')
        Which columns should be expanded:
        - "auto": All non-numeric columns will be expanded.
        - None: no columns will be expanded.
        - list[str]: list of column names to expand.
    dummy_na : {None, False, 'all', 'expanded'}, (default: 'all')
        Options for adding indicator columns for missing values:
        - None or False: do not add indicator columns for missing values
        - 'all': add indicator columns for all columns with missing values
          in fit data
        - 'expanded': add indicator columns for all categorically expanded
          columns (matches `True` behavior from version 0.1)
    fill_value : {float, np.nan}, (default: 0.0)
        The value to fill for missing values with in expanded columns.
        Can be a float or np.nan.
    dataframe_output : bool (default: False)
        If True, ETL output is a pd.DataFrame instead of a np.Array.
    check_null_cols : {None, False, 'raise', 'warn'} (default: False)
        How columns of all nulls should be handled:
        - None or False: do not check for null columns (best performance
          during `fit`).
        - 'raise': raises a RuntimeError if null columns are found
        - 'warn': issues a warning if null columns are found

    Attributes
    ----------
    levels_, dict
        Dictionary created which stores the transformation. Each key is a
        column and each value is an ordered list of values corresponding
        to the categorical representation of the column in question.
    required_columns_, list[str]
        List of column names without cols_to_drop, but before expanding
        cols_to_expand. This is the minimal list of columns required to
        be in the input data when .transform is called.
    columns_, list[str]
        List of final column names in order
    """
    expansion_warn_threshold = 500  # Warn when expanding this many categories
    expansion_exc_threshold = 5000  # Error when expanding this many categories

    def __init__(self,
                 cols_to_drop=None,
                 cols_to_expand='auto',
                 dummy_na='all',
                 fill_value=0.0,
                 dataframe_output=False,
                 check_null_cols=False):
        self.cols_to_drop = cols_to_drop
        self.cols_to_expand = cols_to_expand
        if dummy_na is True:
            warnings.warn(
                '`True` option for dummy_na is deprecated, use '
                '"all" or "expanded" instead',
                DeprecationWarning
            )
        self.dummy_na = dummy_na
        self.fill_value = fill_value
        self.dataframe_output = dataframe_output
        self.check_null_cols = check_null_cols

    def _flag_nulls(self, X, cols_to_drop):
        null_cols = [col for col in X if
                     col not in cols_to_drop and
                     pd.isnull(X[col].values[0]) and
                     X[col].first_valid_index() is None]
        if len(null_cols) > 0:
            if self.check_null_cols == 'warn':
                warnings.warn('The following columns contain only nulls '
                              'and will be dropped: ' + str(null_cols),
                              UserWarning)
            elif self.check_null_cols == 'raise':
                raise RuntimeError('The following columns contain only '
                                   'nulls: ' + str(null_cols))
            else:
                raise ValueError('DataFrameETL.check_null_cols must be '
                                 'one of the following: [None, False, '
                                 '"raise", or "warn"]')
        return cols_to_drop + null_cols

    def _check_sentinels(self, X):
        """Replace default sentinel with random values if the default appears
        in the data."""
        vals = chain.from_iterable(pd.unique(X[col]) for
                                   col in self._cols_to_expand)

        while any(u == self._nan_sentinel for u in vals):
            self._nan_sentinel = uuid.uuid4().hex

    def _create_levels(self, X):
        """Create levels for each column in cols_to_expand."""
        levels = {}
        warn_list = {}
        error_list = {}
        # get a list of categories when the column is cast to
        # dtype category
        # levels are sorted by default
        for col in self._cols_to_expand:
            levels[col] = X[col].astype('category').cat.categories.tolist()
            if (self.expansion_warn_threshold and
                    len(levels[col]) >= self.expansion_warn_threshold):
                warn_list[col] = len(levels[col])
            if (self.expansion_exc_threshold and
                    len(levels[col]) >= self.expansion_exc_threshold):
                error_list[col] = len(levels[col])
            # if there are nans, we will be replacing them with a sentinel,
            # so add the sentinel as a level explicitly
            # Note that even if we don't include a dummy_na column, we still
            # need to keep track of missing values internally for fill_value
            if self._dummy_na == 'expanded' or any(X[col].isnull()):
                levels[col].extend([self._nan_sentinel])
        log.debug("Categories (including nulls) for each column: %s",
                  "; ".join('"%s": %d' % (c, len(l))
                            for c, l in levels.items()))

        if warn_list:
            warnings.warn("The following categorical column(s) have a large "
                          "number of categories. Are you sure you wish to "
                          "convert them to binary indicators?\n%s" %
                          ("; ".join(['"%s": %d categories' % (c, l)
                                      for c, l in warn_list.items()])),
                          RuntimeWarning)
        if error_list:
            err = ("The following column(s) have a very large number of "
                   "categories and may use up too much memory if expanded. "
                   "If you are sure you want to expand these features, "
                   "manually update the expansion_exc_threshold attribute "
                   "and rerun. Otherwise, exclude these columns from "
                   "categorical expansion.\n%s" %
                   ("; ".join(['"%s": %d categories' % (c, l)
                               for c, l in error_list.items()])))
            raise RuntimeError(err)
        return levels

    def _flag_unexpanded_nans(self, X):
        """Optionally create levels for columns we don't want to expand,
        but have nulls."""
        unexpanded_nans = {}
        skip_cols = set(self._cols_to_drop + self._cols_to_expand)
        for col in X.columns:
            if col not in skip_cols:
                if self._dummy_na == 'all' and any(X[col].isnull()):
                    unexpanded_nans[col] = True
                else:
                    unexpanded_nans[col] = False
        return unexpanded_nans

    def _create_col_names(self, X):
        """Identify levels and build an ordered list of final column names."""
        # get unexpanded columns, remove any that need dropping
        unexpanded_cnames = [col for col in X.columns.tolist() if
                             col not in self._cols_to_drop]

        # replace unexpanded colnames with a list of expanded colnames,
        # to maintain column order
        cnames = []
        for col in unexpanded_cnames:
            if col in self._cols_to_expand:
                col_levels = self.levels_[col]
                if self._dummy_na in ['expanded', 'all']:
                    # avoid exposing the sentinel to the user by replacing
                    # it with 'NaN'. If 'NaN' is already a level, use the
                    # sentinel to prevent column name duplicates.
                    # Note that for "expanded", all expanded features will have
                    # a dummied NaN column, which for "all", features with
                    # nulls (expanded or not) will have a dummied NaN column.
                    if 'NaN' in col_levels:
                        expanded_names = ['%s_%s' % (col, self._nan_sentinel)
                                          if cat == self._nan_sentinel
                                          else '%s_%s' % (col, cat) for cat in
                                          col_levels]
                    else:
                        expanded_names = ['%s_NaN' % (col)
                                          if cat == self._nan_sentinel
                                          else '%s_%s' % (col, cat) for cat in
                                          col_levels]
                else:
                    # if the final data frame will not have a dummy na column,
                    # don't include it in the unexpanded list.
                    expanded_names = ['%s_%s' % (col, cat) for cat in
                                      col_levels if cat != self._nan_sentinel]
                cnames.extend(expanded_names)
            else:
                cnames.append(col)
                # Add columns for nulls in unexpanded columns
                if self._unexpanded_nans[col]:
                    cnames.append('%s_NaN' % (col))
        return cnames, unexpanded_cnames

    def _expand_col(self, X, col):
        """Perform categorical expansion on a single column."""
        # Convert the input to a categorical and fill missing
        # values with our sentinel. If the input has sentinels already,
        # then it's a category not seen during the fit and should
        # be ignored in the expansion.
        catcol = X[col].astype('category')

        # check for overlap between categories in col and original fit levels
        overlap = len([lvl for lvl in catcol.cat.categories if
                       lvl in self.levels_[col]])
        if overlap == 0:
            warn_msg = "No overlap between levels in column " + \
                       "'%s' and levels seen during fit" % (col)
            warnings.warn(warn_msg, UserWarning)

        if self._nan_sentinel not in catcol.cat.categories:
            catcol = catcol.cat.add_categories(self._nan_sentinel)
            sentinel_entries = None
        else:
            sentinel_entries = (catcol == self._nan_sentinel)
        catcol = catcol.fillna(self._nan_sentinel)

        # One-hot-encode the array, using the levels seen during fit.
        # When we expand, the only missing values will be categories
        # not seen during fit. We ignore those when encoding
        # by using `dummy_na=False`.
        newcat = catcol.cat.set_categories(self.levels_[col])
        expanded_array = pd.get_dummies(newcat, dummy_na=False)
        expanded_array = expanded_array.values.astype('float32')

        is_nan = (newcat == self._nan_sentinel)
        if is_nan.any():
            expanded_array[is_nan, :-1] = self.fill_value
            if not self._dummy_na:
                # Drop the last column, which is the nan column
                expanded_array = expanded_array[:, :-1]

        # replace sentinel entries which were treated like nans to
        # 0 in the final array
        if sentinel_entries is not None:
            expanded_array[sentinel_entries, :] = 0.0
        return expanded_array

    def fit(self, X, y=None):
        """Fit the ETL pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Training or test data.
        y : numpy array of shape [n_samples]
            Ignored by this class.

        Returns
        -------
        self : returns an instance of DataFrameETL
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ETL transformer must be fit to a dataframe.")

        # TODO: remove in version 1.0.0
        if self.dummy_na is True:
            self._dummy_na = 'expanded'
        else:
            self._dummy_na = self.dummy_na

        # check that a valid dummy_na value passed in
        valid_options = [None, False, 'all', 'expanded']
        if self._dummy_na not in valid_options:
            raise ValueError('dummy_na must be one of %s' % valid_options)

        # set default values
        #  _nan_sentinel is a temporary sentinel for np.nan values
        self._nan_sentinel = 'NaN_Sentinel'
        self.levels_ = {}
        if self.cols_to_drop is None:
            self._cols_to_drop = []
        else:
            self._cols_to_drop = self.cols_to_drop
        # Remove any columns which are all np.nan
        if self.check_null_cols:
            self._cols_to_drop = self._flag_nulls(X, self._cols_to_drop)

        # If None, skip fit step, since we won't do any expansion
        if self.cols_to_expand is None:
            self._cols_to_expand = []
        else:
            if self.cols_to_expand == 'auto':
                self._cols_to_expand = X.columns[
                    (X.dtypes == 'object') | (X.dtypes == 'category')].tolist()
            else:
                self._cols_to_expand = [c for c in self.cols_to_expand if
                                        c in X.columns]
            self._cols_to_expand = [c for c in self._cols_to_expand if
                                    c not in self._cols_to_drop]
            log.debug("There are %d column(s) to expand.",
                      len(self._cols_to_expand))
            # Update sentinels if the defaults are in the dataframe
            self._check_sentinels(X)
            self.levels_ = self._create_levels(X)

        # optionally flag unexpanded columns with nans
        self._unexpanded_nans = self._flag_unexpanded_nans(X)

        # Get colummn names in order
        self.columns_, self.required_columns_ = self._create_col_names(X)

        return self

    def transform(self, X):
        """Create an expanded array or DataFrame and fill it with
        transformed data in chunks, for memory efficiency.

        Parameters
        ----------
        X : pd.DataFrame
            Training or test data.

        Returns
        -------
        np.Array or pd.DataFrame
            Original dataframe with dropped columns and expansion of
            categorical variables, converted to an array or dataframe
            of float32s.
        """

        # If columns_ attribute isn't present, it hasn't been fitted
        if not hasattr(self, 'columns_'):
            raise NotFittedError('This DataFrameETL instance is '
                                 'not fitted yet',)

        log.debug("The input data for transformation have shape %s. "
                  "They will be expanded to shape (%d, %d).",
                  str(X.shape), X.shape[0], len(self.columns_))
        if self.dataframe_output:
            # preallocate a dataframe
            X_new = pd.DataFrame(index=X.index, columns=self.columns_)
            # column index
            i = 0
            for col in self.required_columns_:
                if col in self._cols_to_expand:
                    # assigning this to a temp variable so we can
                    # figure out its shape
                    expanded = self._expand_col(X, col)
                    # this loop is to prevent type coercions from
                    # assigning values to multiple pandas columns
                    # at once
                    for j in range(expanded.shape[1]):
                        X_new.iloc[:, i + j] = expanded[:, j]
                    i += expanded.shape[1]
                else:
                    # put the column in the array
                    X_new.iloc[:, i] = X[col].astype('float32')
                    i += 1
                    if self._unexpanded_nans[col]:
                        X_new.iloc[:, i] = X[col].isnull().astype('float32')
                        i += 1
        else:
            # preallocate an array
            ncol = len(self.columns_)
            X_new = np.empty([X.shape[0], ncol], dtype='float32')
            # column index
            i = 0
            for col in self.required_columns_:
                if col in self._cols_to_expand:
                    # assigning this to a temp variable so we can
                    # figure out its shape
                    expanded = self._expand_col(X, col)
                    X_new[:, i:(i + expanded.shape[1])] = expanded
                    i += expanded.shape[1]
                else:
                    # put the column in the array
                    X_new[:, i] = X[col].astype('float32')
                    i += 1
                    if self._unexpanded_nans[col]:
                        X_new[:, i] = X[col].isnull().astype('float32')
                        i += 1

        return X_new
