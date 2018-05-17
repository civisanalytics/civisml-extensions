# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased


## [0.1.9] - 2018-05-17
### Fixed
- In ``DataFrameETL``, don't check for levels to expand in columns which
  are slated to be dropped. This will avoid raising a warning for too
  many levels in a column if the user has intentionally excluded
  that column (#39).

## [0.1.8] - 2018-04-19
### Fixed
- Fixed ``DataFrameETL`` transformations of ``DataFrame``s with non-trivial
  index when preserving ``DataFrame`` output type (#32, #33)
- Add ``pandas`` version restrictions by Python version (#37)
- Fix code which was incompatible with older ``pandas`` version (#37)

## [0.1.7] - 2018-03-27
### Added
- Added debug log emits for the ``DataFrameETL`` transformer (#24, #27)
- Added debug log emits for the ``HyperbandSearchCV`` estimator (#28, #29)
- Emit a warning if the user attempts to expand a column with
  too many categories (#25, #26)

## [0.1.6] - 2018-1-12

### Fixed
- Now caching CV indices. When CV generators are passed with `shuffle=True` and
  no `random_state` is set, they produce different CV folds on each call to
  `split` (#22).
- Updated `scipy` dependency in `requirements.txt` file to `scipy>=0.14,<2.0`
- ``DataFrameETL`` now correctly handles all ``Categorial``-type columns
  in input ``DataFrame``s. The fix also improves execution time of
  ``transform`` calls by 2-3x (#20).

## [0.1.5] - 2017-10-27

### Added
- Added `check_null_cols` argument to check for null columns (#13)

## [0.1.4] - 2017-10-11

### Fixed
- Fixed bug with fit_params handling in stacking (#12)

## [0.1.3] - 2017-10-5

### Fixed
- Resolved issues with one and two-level edge cases for categorical
  expansion (#10)

## [0.1.2] - 2017-10-3

### Fixed
- Included `y=None` in the fit method definition of DataFrameETL (#7)

### Changed
- Improved parallel performance for hyperband (#8)

## [0.1.1] - 2017-09-13

### Fixed
- Fixed version requirements for scikit-learn to properly import `MaskedArray` (#4).
- In the stacking estimators, get_params no longer throws index error
  when `estimator_list` is an empty list (#6).

## [0.1.0] - 2017-09-12

### Added
- initial commit
