# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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
