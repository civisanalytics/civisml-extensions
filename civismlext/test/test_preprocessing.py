from __future__ import print_function
from __future__ import division

import pickle
import io
import warnings

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import (
    check_transformer_data_not_an_array,
    check_transformer_general,
    check_transformers_unfitted,
    check_transformer_n_iter,
    check_fit2d_predict1d,
    check_fit2d_1sample,
    check_fit2d_1feature,
    check_fit1d_1feature,
    check_fit1d_1sample,
    check_get_params_invariance,
    check_dict_unchanged,
    check_dont_overwrite_parameters,
    check_parameters_default_constructible,
    check_no_fit_attributes_set_in_init)
import pytest

from civismlext.preprocessing import DataFrameETL


NAN_STRING = 'NaN_Sentinel'
NAN_NUMERIC = -99999.0


@pytest.fixture()
def data_raw():
    raw = pd.concat([
        pd.Series(['a', 'b', 'c'], dtype='category', name='pid'),
        pd.Series(['marid', 'effrit', 'sila'], dtype='object',
                  name='djinn_type'),
        pd.Series([1.0, np.NaN, 3.0], dtype='float', name='fruits'),
        pd.Series([2000, 2500, 3000], dtype='uint16', name='age'),
        pd.Series(['cat', 'dog', np.NaN], dtype='category', name='animal'),
    ], axis=1)
    return raw


@pytest.fixture()
def dataframe_expected():
    df = pd.concat([
        pd.Series([0., 1., 0.], dtype='float32', name='djinn_type_effrit'),
        pd.Series([1., 0., 0.], dtype='float32', name='djinn_type_marid'),
        pd.Series([0., 0., 1.], dtype='float32', name='djinn_type_sila'),
        pd.Series([0., 0., 0.], dtype='float32', name='djinn_type_NaN'),
        pd.Series([1., 0., 0.], dtype='float32', name='fruits_1.0'),
        pd.Series([0., 0., 1.], dtype='float32', name='fruits_3.0'),
        pd.Series([0., 1., 0.], dtype='float32', name='fruits_NaN'),
        pd.Series([2000., 2500., 3000.], dtype='float32', name='age'),
        pd.Series([1., 0., 0.], dtype='float32', name='animal_cat'),
        pd.Series([0., 1., 0.], dtype='float32', name='animal_dog'),
        pd.Series([0., 0., 1.], dtype='float32', name='animal_NaN'),
    ], axis=1)
    return df


@pytest.fixture()
def data_raw_2():
    raw = pd.concat([
        pd.Series(['c', np.NaN, 'NaN_Sentinel'], dtype='category', name='pid'),
        pd.Series(['marid', 'effrit', 'sila'], dtype='object',
                  name='djinn_type'),
        pd.Series([-99999.0, np.NaN, 1.0], dtype='float', name='fruits'),
        pd.Series([2000, 2500, 3000], dtype='uint16', name='age'),
        pd.Series(['cat', 'dog', np.NaN], dtype='category', name='animal'),
    ], axis=1)
    return raw


@pytest.fixture()
def dataframe_2_expected():
    df = pd.concat([
        pd.Series([0., 0., 0.], dtype='float32', name='pid_a'),
        pd.Series([0., 0., 0.], dtype='float32', name='pid_b'),
        pd.Series([1., 0., 0.], dtype='float32', name='pid_c'),
        pd.Series([0., 1., 0.], dtype='float32', name='pid_NaN'),
        pd.Series([0., 0., 1.], dtype='float32', name='fruits_1.0'),
        pd.Series([0., 0., 0.], dtype='float32', name='fruits_3.0'),
        pd.Series([0., 1., 0.], dtype='float32', name='fruits_NaN')
    ], axis=1)
    return df


@pytest.fixture()
def data_few_levels():
    raw = pd.concat([
        pd.Series(['a', 'NaN', 'NaN'], dtype='category', name='pid'),
        pd.Series([1.0, np.NaN, 1.0], dtype='float', name='fruits'),
        pd.Series(['cat', 'cat', 'cat'], dtype='category', name='animal'),
    ], axis=1)
    return raw


@pytest.fixture()
def few_levels_expected():
    df = pd.concat([
        pd.Series([0., 1., 1.], dtype='float32', name='pid_NaN'),
        pd.Series([1., 0., 0.], dtype='float32', name='pid_a'),
        pd.Series([0., 0., 0.], dtype='float32', name='pid_NaN_Sentinel'),
        pd.Series([1., 99., 1.], dtype='float32', name='fruits_1.0'),
        pd.Series([0., 1., 0.], dtype='float32', name='fruits_NaN'),
        pd.Series([1., 1., 1.], dtype='float32', name='animal_cat'),
        pd.Series([0., 0., 0.], dtype='float32', name='animal_NaN')
    ], axis=1)
    return df


@pytest.fixture()
def levels_dict():
    levels = {
        'pid': ['a', 'b', 'c', NAN_STRING],
        'djinn_type': ['effrit', 'marid', 'sila', NAN_STRING],
        'animal': ['cat', 'dog', NAN_STRING]
    }
    return levels


@pytest.fixture()
def levels_dict_numeric():
    levels = {'pid': ['a', 'b', 'c', NAN_STRING],
              'fruits': [1.0, 3.0, NAN_NUMERIC]}
    return levels


def test_sklearn_api():
    name = DataFrameETL.__name__
    check_parameters_default_constructible(name, DataFrameETL)
    check_no_fit_attributes_set_in_init(name, DataFrameETL)

    estimator = DataFrameETL()

    for check in [
            check_transformers_unfitted,
            check_transformer_n_iter,
            check_get_params_invariance]:
        check(name, estimator)

    # these are known failures
    for check in [
            check_transformer_data_not_an_array,
            check_transformer_general,
            check_fit2d_predict1d,
            check_fit2d_1sample,
            check_fit2d_1feature,
            check_fit1d_1feature,
            check_fit1d_1sample,
            check_dict_unchanged,
            check_dont_overwrite_parameters]:
        with pytest.raises(TypeError) as e:
            check(name, estimator)
        assert (
            "ETL transformer must be fit "
            "to a dataframe.") in str(e.value)


def test_flag_nulls_warn(data_raw):
    expander = DataFrameETL(check_null_cols='warn')
    drop_cols = ['col1', 'col2']
    drop_cols_2 = ['col1', 'col2', 'nantastic']
    assert expander._flag_nulls(data_raw, []) == []
    assert expander._flag_nulls(data_raw, drop_cols) == drop_cols

    # add a col of all nans
    data_raw['nantastic'] = pd.Series([np.NaN] * 3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # check that we don't add the col if it's already being dropped
        assert expander._flag_nulls(data_raw, drop_cols_2) == drop_cols_2
        assert len(w) == 0

        assert expander._flag_nulls(data_raw, drop_cols) == drop_cols_2
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)


def test_flag_nulls_raise(data_raw):
    expander = DataFrameETL(check_null_cols='raise')
    drop_cols = ['col1', 'col2']

    # add a col of all nans
    data_raw['nantastic'] = pd.Series([np.NaN] * 3)
    with pytest.raises(RuntimeError):
        expander._flag_nulls(data_raw, drop_cols)


def test_flag_numeric():
    test1 = [1, 'a', 'b']
    test2 = [5.55, 0, np.nan]
    expander = DataFrameETL()
    assert expander._flag_numeric(test1) is False
    assert expander._flag_numeric(test2) is True


def test_check_sentinels(data_raw):
    expander = DataFrameETL(cols_to_expand=['pid', 'djinn_type',
                                            'animal', 'fruits'])
    # fill in necessary parameters
    expander._nan_string = 'effrit'
    expander._nan_numeric = 1.0
    expander._is_numeric = {}
    expander.levels_ = {}
    expander._cols_to_drop = expander.cols_to_drop
    expander._cols_to_expand = expander.cols_to_expand
    for col in expander.cols_to_expand:
        expander._is_numeric[col] = expander._flag_numeric(
            pd.unique(data_raw[col]))
    expander._check_sentinels(data_raw)
    assert expander._nan_string is not 'effrit'
    assert expander._nan_numeric is not 1.0
    assert not (data_raw[['pid', 'djinn_type', 'animal']] ==
                expander._nan_string).any().any()
    assert not (data_raw['fruits'] == expander._nan_numeric).any().any()


def test_create_levels(data_raw, levels_dict):
    expander = DataFrameETL(cols_to_expand=['pid', 'djinn_type', 'animal'])
    expander._is_numeric = {'pid': 0, 'djinn_type': 0, 'animal': 0}
    expander._nan_numeric = NAN_NUMERIC
    expander._nan_string = NAN_STRING
    expander._cols_to_drop = expander.cols_to_drop
    expander._cols_to_expand = expander.cols_to_expand
    actual_levels = expander._create_levels(data_raw)
    assert actual_levels == levels_dict


def test_create_levels_no_dummy(data_raw, levels_dict_numeric):
    expander = DataFrameETL(cols_to_expand=['pid', 'fruits'],
                            dummy_na=False)
    # remove nan from pid levels
    levels_dict_numeric['pid'] = ['a', 'b', 'c']
    expander._is_numeric = {'pid': 0, 'fruits': 1}
    expander._nan_numeric = NAN_NUMERIC
    expander._nan_string = NAN_STRING
    expander._cols_to_drop = expander.cols_to_drop
    expander._cols_to_expand = expander.cols_to_expand
    actual_levels = expander._create_levels(data_raw)
    assert actual_levels == levels_dict_numeric


def test_create_col_names(data_raw):
    expander = DataFrameETL(cols_to_expand=['pid', 'djinn_type', 'animal'],
                            cols_to_drop=['fruits'],
                            dummy_na=True)
    expander._is_numeric = {'pid': 0, 'djinn_type': 0, 'animal': 0}
    expander._nan_numeric = NAN_NUMERIC
    expander._nan_string = NAN_STRING
    expander._cols_to_drop = expander.cols_to_drop
    expander._cols_to_expand = expander.cols_to_expand
    expander.levels_ = expander._create_levels(data_raw)
    (cnames, unexpanded) = expander._create_col_names(data_raw)
    cols_expected = ['pid_a', 'pid_b', 'pid_c', 'pid_NaN',
                     'djinn_type_effrit', 'djinn_type_marid',
                     'djinn_type_sila', 'djinn_type_NaN', 'age',
                     'animal_cat', 'animal_dog', 'animal_NaN']
    assert cnames == cols_expected
    assert unexpanded == ['pid', 'djinn_type', 'age', 'animal']


def test_create_col_names_no_dummy(data_raw):
    expander = DataFrameETL(cols_to_expand=['pid', 'djinn_type', 'animal'],
                            cols_to_drop=['fruits'],
                            dummy_na=False)
    expander._is_numeric = {'pid': 0, 'djinn_type': 0, 'animal': 0}
    expander._nan_numeric = NAN_NUMERIC
    expander._nan_string = NAN_STRING
    expander._cols_to_drop = expander.cols_to_drop
    expander._cols_to_expand = expander.cols_to_expand
    expander.levels_ = expander._create_levels(data_raw)
    (cnames, unexpanded) = expander._create_col_names(data_raw)
    cols_expected = ['pid_a', 'pid_b', 'pid_c',
                     'djinn_type_effrit', 'djinn_type_marid',
                     'djinn_type_sila', 'age',
                     'animal_cat', 'animal_dog']
    assert cnames == cols_expected
    assert unexpanded == ['pid', 'djinn_type', 'age', 'animal']


def test_create_col_names_numeric(data_raw):
    expander = DataFrameETL(cols_to_expand=['pid', 'fruits'],
                            cols_to_drop=['djinn_type', 'animal'],
                            dummy_na=True)
    expander._is_numeric = {'pid': 0, 'djinn_type': 0, 'fruits': 0}
    expander._nan_numeric = NAN_NUMERIC
    expander._nan_string = NAN_STRING
    expander._cols_to_drop = expander.cols_to_drop
    expander._cols_to_expand = expander.cols_to_expand
    expander.levels_ = expander._create_levels(data_raw)
    (cnames, unexpanded) = expander._create_col_names(data_raw)
    cols_numeric = ['pid_a', 'pid_b', 'pid_c', 'pid_NaN', 'fruits_1.0',
                    'fruits_3.0', 'fruits_NaN', 'age']
    assert cnames == cols_numeric
    assert unexpanded == ['pid', 'fruits', 'age']


def test_add_sentinel(data_raw):
    expander = DataFrameETL()
    expander._is_numeric = {'pid': 0, 'djinn_type': 0, 'animal': 0,
                            'fruits': 1, 'age': 1}
    expander._nan_numeric = NAN_NUMERIC
    expander._nan_string = NAN_STRING
    # this shouldn't add any sentinels
    col = expander._add_sentinel('age', data_raw['age'])
    pd.testing.assert_series_equal(col, data_raw['age'].astype('uint16'))
    # this should add a sentinel
    col2 = expander._add_sentinel('animal', data_raw['animal'])
    pd.testing.assert_series_equal(col2,
                                   pd.Series(['cat', 'dog', NAN_STRING],
                                             dtype='object', name='animal'))
    # this should add a numeric sentinel
    col2 = expander._add_sentinel('fruits', data_raw['fruits'])
    pd.testing.assert_series_equal(col2,
                                   pd.Series([1.0, NAN_NUMERIC, 3.0],
                                             dtype='float', name='fruits'))


def test_expand_col(data_raw):
    expander = DataFrameETL(cols_to_drop=['fruits'],
                            dummy_na=True,
                            fill_value=-1.0)
    expander.fit(data_raw)
    # should expand even if there are no NaNs
    arr = expander._expand_col(data_raw, 'pid')
    arr_exp = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                        [0., 0., 1., 0.]])
    assert_almost_equal(arr, arr_exp)
    arr = expander._expand_col(data_raw, 'animal')
    arr_exp = np.array([[1., 0., 0.], [0., 1., 0.], [-1., -1., 1.]])
    assert_almost_equal(arr, arr_exp)


def test_expand_col_no_dummy(data_raw):
    expander = DataFrameETL(cols_to_drop=['fruits'],
                            dummy_na=False,
                            fill_value=-1.0)
    expander.fit(data_raw)
    arr = expander._expand_col(data_raw, 'pid')
    arr_exp = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    assert_almost_equal(arr, arr_exp)
    arr = expander._expand_col(data_raw, 'animal')
    arr_exp = np.array([[1., 0.], [0., 1.], [-1., -1.]])
    assert_almost_equal(arr, arr_exp)


def test_expand_col_numeric(data_raw):
    expander = DataFrameETL(cols_to_drop=['pid', 'djinn_type', 'animal'],
                            cols_to_expand=['fruits'],
                            dummy_na=True,
                            fill_value=0.0)
    expander.fit(data_raw)
    arr = expander._expand_col(data_raw, 'fruits')
    arr_exp = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    assert_almost_equal(arr, arr_exp)


def test_expand_col_numeric_no_dummy(data_raw):
    expander = DataFrameETL(cols_to_drop=['pid', 'djinn_type', 'animal'],
                            cols_to_expand=['fruits'],
                            dummy_na=False,
                            fill_value=np.nan)
    expander.fit(data_raw)
    arr = expander._expand_col(data_raw, 'fruits')
    arr_exp = np.array([[1., 0.], [np.nan, np.nan], [0., 1.]])
    assert_almost_equal(arr, arr_exp)


def test_expand_col_few_levels(data_few_levels, few_levels_expected):
    expander = DataFrameETL(cols_to_expand=['pid', 'fruits', 'animal'],
                            dummy_na=True,
                            fill_value=99.)
    expander.fit(data_few_levels)
    arr = expander._expand_col(data_few_levels, 'pid')
    expected_array = np.asarray(few_levels_expected[['pid_NaN', 'pid_a',
                                                     'pid_NaN_Sentinel']])
    assert_almost_equal(arr, expected_array)

    arr = expander._expand_col(data_few_levels, 'fruits')
    expected_array = np.asarray(few_levels_expected[['fruits_1.0',
                                                     'fruits_NaN']])
    assert_almost_equal(arr, expected_array)

    arr = expander._expand_col(data_few_levels, 'animal')
    expected_array = np.asarray(few_levels_expected[['animal_cat',
                                                     'animal_NaN']])
    assert_almost_equal(arr, expected_array)


def test_expand_col_few_levels_no_dummy(data_few_levels, few_levels_expected):
    expander = DataFrameETL(cols_to_expand=['pid', 'fruits', 'animal'],
                            dummy_na=False,
                            fill_value=99.)
    expander.fit(data_few_levels)
    arr = expander._expand_col(data_few_levels, 'pid')
    expected_array = np.asarray(few_levels_expected[['pid_NaN', 'pid_a']])
    assert_almost_equal(arr, expected_array)

    arr = expander._expand_col(data_few_levels, 'fruits')
    expected_array = np.asarray(few_levels_expected[['fruits_1.0']])
    assert_almost_equal(arr, expected_array)

    arr = expander._expand_col(data_few_levels, 'animal')
    expected_array = np.asarray(few_levels_expected[['animal_cat']])
    assert_almost_equal(arr, expected_array)


def test_fit_exc(dataframe_expected):
    expander = DataFrameETL()
    arr = np.asarray(dataframe_expected)
    with pytest.raises(TypeError):
        expander.fit(arr)


def test_fit_auto(data_raw, levels_dict):
    # test auto identification of columns to expand
    expander = DataFrameETL(cols_to_drop=['fruits'],
                            cols_to_expand='auto',
                            dummy_na=True)
    expander.fit(data_raw)
    cols_expected = ['pid_a', 'pid_b', 'pid_c', 'pid_NaN',
                     'djinn_type_effrit', 'djinn_type_marid',
                     'djinn_type_sila', 'djinn_type_NaN', 'age',
                     'animal_cat', 'animal_dog', 'animal_NaN']
    assert expander.levels_ == levels_dict
    assert expander.columns_ == cols_expected
    assert expander.required_columns_ == ['pid', 'djinn_type',
                                          'age', 'animal']


def test_fit_none(data_raw):
    # No dropping, no expansion
    expander = DataFrameETL(cols_to_drop=None,
                            cols_to_expand=None)
    expander.fit(data_raw)
    # column names should be identical to dataframe names
    data_cnames = data_raw.columns.tolist()
    assert expander.columns_ == data_cnames
    assert expander.required_columns_ == data_cnames


def test_fit_drop_only(data_raw):
    # test dropping columns but not expanding them
    expander = DataFrameETL(cols_to_drop=['fruits'],
                            cols_to_expand=None)
    expander.fit(data_raw)
    # both lists will be the same, since no cols were expanded
    assert expander.columns_ == ['pid', 'djinn_type', 'age', 'animal']
    assert expander.required_columns_ == ['pid', 'djinn_type',
                                          'age', 'animal']


def test_fit_list(data_raw, levels_dict):
    # test that fit handles list of cols to expand correctly
    expander = DataFrameETL(cols_to_drop=['fruits'],
                            cols_to_expand=['pid', 'djinn_type', 'animal'],
                            dummy_na=True)
    expander.fit(data_raw)
    cols_expected = ['pid_a', 'pid_b', 'pid_c', 'pid_NaN',
                     'djinn_type_effrit', 'djinn_type_marid',
                     'djinn_type_sila', 'djinn_type_NaN', 'age',
                     'animal_cat', 'animal_dog', 'animal_NaN']
    assert expander.levels_ == levels_dict
    assert expander.columns_ == cols_expected
    assert expander.required_columns_ == ['pid', 'djinn_type',
                                          'age', 'animal']


def test_fit_list_numeric(data_raw, levels_dict_numeric):
    expander = DataFrameETL(cols_to_drop=['djinn_type', 'animal'],
                            cols_to_expand=['pid', 'fruits'],
                            dummy_na=True)
    expander.fit(data_raw)
    assert expander.levels_ == levels_dict_numeric
    cols_numeric = ['pid_a', 'pid_b', 'pid_c', 'pid_NaN', 'fruits_1.0',
                    'fruits_3.0', 'fruits_NaN', 'age']
    assert expander.columns_ == cols_numeric
    assert expander.required_columns_ == ['pid', 'fruits', 'age']


def test_fit_with_nan_col(data_raw, levels_dict):
    # test that fit handles all-nan columns correctly
    data_raw['nantastic'] = pd.Series([np.NaN] * 3)
    expander = DataFrameETL(cols_to_drop=['fruits'],
                            cols_to_expand=['pid', 'djinn_type', 'animal'],
                            dummy_na=True,
                            check_null_cols='warn')
    with warnings.catch_warnings(record=True) as fit_w:
        expander.fit(data_raw)
        cols_expected = ['pid_a', 'pid_b', 'pid_c', 'pid_NaN',
                         'djinn_type_effrit', 'djinn_type_marid',
                         'djinn_type_sila', 'djinn_type_NaN', 'age',
                         'animal_cat', 'animal_dog', 'animal_NaN']
        assert expander.levels_ == levels_dict
        assert expander.columns_ == cols_expected
        assert expander.required_columns_ == ['pid', 'djinn_type',
                                              'age', 'animal']
        assert len(fit_w) == 1
        assert issubclass(fit_w[-1].category, UserWarning)
        assert expander._cols_to_drop == ['fruits', 'nantastic']

    with pytest.raises(RuntimeError):
        expander.check_null_cols = 'raise'
        expander.fit(data_raw)


def test_transform_notfitted(data_raw):
    expander = DataFrameETL()
    with pytest.raises(NotFittedError):
        expander.transform(data_raw)


def test_transform(data_raw, dataframe_expected):
    expander = DataFrameETL(cols_to_drop=['pid'],
                            cols_to_expand=['djinn_type', 'fruits', 'animal'],
                            dummy_na=True)
    expander.fit(data_raw)
    arr = expander.transform(data_raw)

    expected_array = np.asarray(dataframe_expected)
    assert arr.shape == expected_array.shape
    assert_almost_equal(arr, expected_array)


def test_transform_no_dummy(data_raw, dataframe_expected):
    expander = DataFrameETL(cols_to_drop=['pid'],
                            cols_to_expand=['djinn_type', 'fruits', 'animal'],
                            dummy_na=False)
    expander.fit(data_raw)
    arr = expander.transform(data_raw)

    # drop nan columns from expected data
    dataframe_expected.pop('djinn_type_NaN')
    dataframe_expected.pop('fruits_NaN')
    dataframe_expected.pop('animal_NaN')
    expected_array = np.asarray(dataframe_expected)
    assert arr.shape == expected_array.shape
    assert_almost_equal(arr, expected_array)


def test_transform_dataframe(data_raw, dataframe_expected):
    expander = DataFrameETL(cols_to_drop=['pid'],
                            cols_to_expand=['djinn_type', 'fruits', 'animal'],
                            dummy_na=True,
                            dataframe_output=True)
    expander.fit(data_raw)
    df = expander.transform(data_raw)
    assert df.shape == dataframe_expected.shape
    assert df.equals(dataframe_expected)


def test_transform_dataframe_no_dummy(data_raw, dataframe_expected):
    expander = DataFrameETL(cols_to_drop=['pid'],
                            cols_to_expand=['djinn_type', 'fruits', 'animal'],
                            dummy_na=False,
                            dataframe_output=True)
    expander.fit(data_raw)
    df = expander.transform(data_raw)

    # drop nan columns from expected data
    dataframe_expected.pop('djinn_type_NaN')
    dataframe_expected.pop('fruits_NaN')
    dataframe_expected.pop('animal_NaN')
    assert df.shape == dataframe_expected.shape
    assert df.equals(dataframe_expected)


def test_transform_two_levels(data_few_levels, few_levels_expected):
    expander = DataFrameETL(cols_to_expand=['pid', 'fruits', 'animal'],
                            dummy_na=True,
                            fill_value=99.,
                            dataframe_output=True)
    expander.fit(data_few_levels)
    df = expander.transform(data_few_levels)
    assert df.shape == few_levels_expected.shape
    assert df.equals(few_levels_expected)


def test_transform_two_levels_no_dummy(data_few_levels, few_levels_expected):
    expander = DataFrameETL(cols_to_expand=['pid', 'fruits', 'animal'],
                            dummy_na=False,
                            fill_value=99.,
                            dataframe_output=True)
    expander.fit(data_few_levels)
    df = expander.transform(data_few_levels)
    few_levels_expected.pop('pid_NaN_Sentinel')
    few_levels_expected.pop('fruits_NaN')
    few_levels_expected.pop('animal_NaN')
    assert df.shape == few_levels_expected.shape
    assert df.equals(few_levels_expected)


def test_transform_reuse_transformer(data_raw, data_raw_2,
                                     dataframe_2_expected):
    expander = DataFrameETL(cols_to_expand=['pid', 'fruits'],
                            cols_to_drop=['djinn_type', 'age', 'animal'],
                            dummy_na=True,
                            dataframe_output=True)
    expander.fit(data_raw)
    df = expander.transform(data_raw_2)
    assert df.equals(dataframe_2_expected)


def test_transform_preserve_col_order(data_raw, data_raw_2,
                                      dataframe_2_expected):
    expander = DataFrameETL(cols_to_expand=['pid', 'fruits'],
                            cols_to_drop=['djinn_type', 'age', 'animal'],
                            dummy_na=True,
                            dataframe_output=True)
    expander.fit(data_raw)
    # swap col order for second data file
    data_raw_2 = data_raw_2[['fruits', 'age', 'djinn_type', 'pid', 'animal']]
    df = expander.transform(data_raw_2)
    assert df.equals(dataframe_2_expected)


def test_transform_bad_expand_col(data_raw, dataframe_expected):
    expander = DataFrameETL(cols_to_drop=['pid'],
                            cols_to_expand=['djinn_type', 'fruits',
                                            'animal', 'not_in_df'],
                            dummy_na=True)
    expander.fit(data_raw)
    arr = expander.transform(data_raw)

    expected_array = np.asarray(dataframe_expected)
    assert arr.shape == expected_array.shape
    assert_almost_equal(arr, expected_array)


def test_refit(data_raw, data_raw_2):
    expander = DataFrameETL(cols_to_drop=['pid', 'age', 'djinn_type'],
                            cols_to_expand=['fruits', 'animal'],
                            dataframe_output=True,
                            dummy_na=True)
    expander.fit(data_raw)
    df = expander.transform(data_raw)
    df_expected = pd.concat([
        pd.Series([1., 0., 0.], dtype='float32', name='fruits_1.0'),
        pd.Series([0., 0., 1.], dtype='float32', name='fruits_3.0'),
        pd.Series([0., 1., 0.], dtype='float32', name='fruits_NaN'),
        pd.Series([1., 0., 0.], dtype='float32', name='animal_cat'),
        pd.Series([0., 1., 0.], dtype='float32', name='animal_dog'),
        pd.Series([0., 0., 1.], dtype='float32', name='animal_NaN'),
    ], axis=1)
    assert df.equals(df_expected)

    expander.fit(data_raw_2)
    df2 = expander.transform(data_raw_2)
    df_expected_2 = pd.concat([
        pd.Series([1., 0., 0.], dtype='float32', name='fruits_-99999.0'),
        pd.Series([0., 0., 1.], dtype='float32', name='fruits_1.0'),
        pd.Series([0., 1., 0.], dtype='float32', name='fruits_NaN'),
        pd.Series([1., 0., 0.], dtype='float32', name='animal_cat'),
        pd.Series([0., 1., 0.], dtype='float32', name='animal_dog'),
        pd.Series([0., 0., 1.], dtype='float32', name='animal_NaN'),
    ], axis=1)
    assert df2.equals(df_expected_2)


def test_pickle(data_raw, dataframe_expected):
    expander = DataFrameETL(cols_to_drop=['pid'],
                            cols_to_expand=['djinn_type', 'fruits', 'animal'],
                            dummy_na=True)
    expander.fit(data_raw)
    # pickle the transformer
    buff = io.BytesIO()
    pickle.dump(expander, buff)
    buff.seek(0)
    # transform data after unpickling transformer
    expander = pickle.load(buff)

    arr = expander.transform(data_raw)
    expected_array = np.asarray(dataframe_expected)
    assert arr.shape == expected_array.shape
    assert_almost_equal(arr, expected_array)
