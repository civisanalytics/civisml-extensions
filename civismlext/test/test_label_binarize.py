import pytest
import numpy as np

from civismlext.preprocessing import _label_binarize


def test_smoke():
    y = ['a', 'b', 'c', 'c']
    classes = ['a', 'b', 'c']
    yenc = _label_binarize(y, classes)
    assert np.array_equal(yenc, np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1]]))


def test_raises_bad_inputs():
    with pytest.raises(ValueError):
        _label_binarize([[]], ['a', 'b'])

    with pytest.raises(ValueError):
        _label_binarize([], ['a', 'b'])

    with pytest.raises(ValueError):
        _label_binarize(np.array([[]]), ['a', 'b'])

    with pytest.raises(ValueError):
        _label_binarize([['a', 'b', 'c']], ['a', 'b'])

    with pytest.raises(ValueError):
        _label_binarize(np.zeros((2, 3)), ['a', 'b'])


def test_one_column_diff_class():
    y = ['a', 'a', 'c']
    classes = ['a']
    yenc = _label_binarize(y, classes)
    assert np.array_equal(yenc, np.array([
        [1],
        [1],
        [0]]))


def test_one_column_same_class():
    y = ['a', 'a', 'a']
    classes = ['a']
    yenc = _label_binarize(y, classes)
    assert np.array_equal(yenc, np.array([
        [1],
        [1],
        [1]]))


def test_two_columns_diff_classes():
    y = ['a', 'c', 'f', 'b']
    classes = ['a', 'd']
    yenc = _label_binarize(y, classes)
    assert np.array_equal(yenc, np.array([
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 0]]))


def test_two_columns_same_classes():
    y = ['a', 'b', 'b']
    classes = ['b', 'a']
    yenc = _label_binarize(y, classes)
    assert np.array_equal(yenc, np.array([
        [0, 1],
        [1, 0],
        [1, 0]]))


def test_columns_diff_classes():
    y = ['a', 'e', 'c', 'b']
    classes = ['b', 'a', 'c', 'f']
    yenc = _label_binarize(y, classes)
    assert np.array_equal(yenc, np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0]]))


def test_columns_same_classes():
    y = ['a', 'd', 'c', 'b']
    classes = ['b', 'a', 'c', 'd']
    yenc = _label_binarize(y, classes)
    assert np.array_equal(yenc, np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0]]))
