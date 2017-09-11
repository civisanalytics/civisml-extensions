civisml-extensions
==================

.. image:: https://www.travis-ci.org/civisanalytics/civisml-extensions.svg?branch=master
    :target: https://www.travis-ci.org/civisanalytics/civisml-extensions

scikit-learn-compatible estimators from Civis Analytics

Installation
------------

Installation with ``pip`` is recommended::

    $ pip install civisml-extensions

For development, a few additional dependencies are needed::

    $ pip install -r dev-requirements.txt

Contents and Usage
------------------

This package contains `scikit-learn`_-compatible estimators for stacking (
``StackedClassifier``, ``StackedRegressor``), non-negative linear regression (
``NonNegativeLinearRegression``), preprocessing pandas_ ``DataFrames`` (
``DataFrameETL``), and using Hyperband_ for cross-validating hyperparameters (
``HyperbandSearchCV``).

Usage of these estimators follows the standard sklearn conventions. Here is an
example of using the ``StackedClassifier``::

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

See the doc strings of the various estimators for more information.

Contributing
------------

See ``CONTIBUTING.md`` for information about contributing to this project.

License
-------

BSD-3

See ``LICENSE.md`` for details.

.. _scikit-learn: http://scikit-learn.org/
.. _pandas: http://pandas.pydata.org/
.. _Hyperband: https://arxiv.org/abs/1603.06560
