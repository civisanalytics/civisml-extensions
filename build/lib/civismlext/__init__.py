import pkg_resources

from civismlext.stacking import StackedRegressor
from civismlext.stacking import StackedClassifier
from civismlext.nonnegative import NonNegativeLinearRegression
from civismlext.hyperband import HyperbandSearchCV
from civismlext.preprocessing import DataFrameETL


__version__ = pkg_resources.get_distribution('civisml-extensions').version

__all__ = [
    'StackedRegressor',
    'StackedClassifier',
    'NonNegativeLinearRegression',
    'HyperbandSearchCV',
    'DataFrameETL',
    '__version__',
]
