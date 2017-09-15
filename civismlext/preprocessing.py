from __future__ import print_function
from __future__ import division

import random
import uuid
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import NotFittedError


class DataFrameETL(BaseEstimator, TransformerMixin):
    """Performs ETL on a dataframe, using the following steps:
    - dropping specified columns
    - performing categorical expansion on a specified set of columns
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
    dummy_na : bool, (default: True)
        Add a column to indicate missing values.
    fill_value : {float, np.nan}, (default: 0.0)
        The value to fill for missing values with in expanded columns.
        Can be a float or np.nan.
    dataframe_output : bool (default: False)
        If True, ETL output is a pd.DataFrame instead of a np.Array.

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
    def __init__(self,
                 cols_to_drop=None,
                 cols_to_expand='auto',
                 dummy_na=True,
                 fill_value=0.0,
                 dataframe_output=False):
        self.cols_to_drop = cols_to_drop
        self.cols_to_expand = cols_to_expand
        self.dummy_na = dummy_na
        self.fill_value = fill_value
        self.dataframe_output = dataframe_output

    def _flag_numeric(self, levels):
        """Duck typing test for if a list is numeric-like."""
        try:
            for level in levels:
                if level is not None:
                    1 + level
            is_numeric = True
        except:
            is_numeric = False
        return is_numeric

    def _check_sentinels(self, X):
        """Replace default sentinels with random values if defaults appear
        in the data."""

        numeric_vals = chain.from_iterable(
            pd.unique(X[col]).tolist() for col in self._cols_to_expand
            if self._is_numeric[col])
        other_vals = chain.from_iterable(
            np.array(pd.unique(X[col])).tolist() for
            col in self._cols_to_expand
            if not self._is_numeric[col])

        while any(u == self._nan_numeric for u in numeric_vals):
            self._nan_numeric = random.randint(0, 1e6)
        while any(u == self._nan_string for u in other_vals):
            self._nan_string = uuid.uuid4().hex

    def _create_levels(self, X):
        """Create levels for each column in cols_to_expand."""
        levels = {}
        # get a list of categories when the column is cast to
        # dtype category
        # levels are sorted by default
        for col in self._cols_to_expand:
            levels[col] = X[col].astype('category').cat.categories.tolist()
            # if there are nans, we will be replacing them with a sentinel,
            # so add the sentinel as a level explicitly
            # Note that even if we don't include a dummy_na column, we still
            # need to keep track of missing values internally for fill_value
            if self.dummy_na or any(X[col].isnull()):
                if self._is_numeric[col]:
                    levels[col].extend([self._nan_numeric])
                else:
                    levels[col].extend([self._nan_string])
        return levels

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
                if self.dummy_na:
                    # avoid exposing the sentinel to the user by replacing
                    # it with 'NaN'. If 'NaN' is already a level, use the
                    # sentinel to prevent column name duplicates.
                    if 'NaN' in col_levels:
                        expanded_names = ['%s_%s' % (col, self._nan_string) if
                                          cat in
                                          [self._nan_string, self._nan_numeric]
                                          else '%s_%s' % (col, cat) for cat in
                                          col_levels]
                    else:
                        expanded_names = ['%s_NaN' % (col) if cat in
                                          [self._nan_string, self._nan_numeric]
                                          else '%s_%s' % (col, cat) for cat in
                                          col_levels]
                else:
                    # if the final data frame will not have a dummy na column,
                    # don't include it in the unexpanded list.
                    expanded_names = ['%s_%s' % (col, cat) for cat in
                                      col_levels if cat not in
                                      [self._nan_string, self._nan_numeric]]
                cnames.extend(expanded_names)
            else:
                cnames.append(col)

        return cnames, unexpanded_cnames

    def _add_sentinel(self, col, nan_col):
        """Add a sentinel for NaN values."""
        if self._is_numeric[col]:
            nan_col = nan_col.fillna(self._nan_numeric)
        else:
            nan_col = nan_col.astype('object').fillna(self._nan_string)
        return nan_col

    def _expand_col(self, X, col):
        """Perform categorical expansion on a single column."""
        # find any values in the data that match the sentinel
        if self._is_numeric[col]:
            sentinel_entries = np.where(np.equal(X[col].values,
                                                 self._nan_numeric))
        else:
            sentinel_entries = np.where(np.equal(X[col].values,
                                                 self._nan_string))

        # replace nans with sentinel value
        noNaN_col = self._add_sentinel(col, X[col])

        # perform categorical expansion
        expanded_array = label_binarize(noNaN_col,
                                        classes=self.levels_[col])

        # if the column has one or two levels, label_binarize
        # returns one column which is !(first level).
        if len(self.levels_[col]) == 1:
            expanded_array = 1 - expanded_array
        elif len(self.levels_[col]) == 2:
            # ensure that number of levels = number of cols,
            # and that order is correct
            expanded_array = np.column_stack([1-expanded_array,
                                              expanded_array])

        ncol = expanded_array.shape[1]
        if self._nan_numeric in self.levels_[col] or \
           self._nan_string in self.levels_[col]:
            # fill in self.fill_value for nan sentinels
            # the nan column will be the last one, because it
            # is appended after other levels are determined
            inds = np.where(expanded_array[:, ncol - 1] == 1)
            expanded_array = expanded_array.astype('float32')
            expanded_array[inds, 0:(ncol-1)] = np.float32(self.fill_value)
            if not self.dummy_na:
                # Drop the last column, which is the nan column
                expanded_array = expanded_array[:, 0:(ncol-1)]
        else:
            expanded_array = expanded_array.astype('float32')

        # replace sentinel entries which were treated like nans to
        # 0 in the final array
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
        # set default values
        #  _nan_string is a temporary sentinel for np.nan values in
        # non-numeric columns.
        # _nan_numeric is a temporary sentinel for np.nan values in numeric
        # columns.
        self._nan_string = 'NaN_Sentinel'
        self._nan_numeric = -99999.0
        # _is_numeric: dictionary of column names and a boolean flag, which
        # is True if the column is numeric
        self._is_numeric = {}
        self.levels_ = {}
        if self.cols_to_drop is None:
            self._cols_to_drop = []
        else:
            self._cols_to_drop = self.cols_to_drop
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
            # Flag for numeric columns
            for col in self._cols_to_expand:
                self._is_numeric[col] = self._flag_numeric(
                    pd.unique(X[col]))
            # Update sentinels if the defaults are in the dataframe
            self._check_sentinels(X)
            self.levels_ = self._create_levels(X)

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

        if self.dataframe_output:
            # preallocate a dataframe
            X_new = pd.DataFrame(index=np.arange(X.shape[0]),
                                 columns=self.columns_)
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

        return X_new
