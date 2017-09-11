# Contributing to sklearn-extensions

We welcome bug reports and pull requests from everyone!
This project is intended to be a safe, welcoming space for collaboration, and
contributors are expected to adhere to the
[Contributor Covenant](http://contributor-covenant.org) code of conduct.


## Getting Started

There are two ways to contribute:

### File an Issue

If you find a bug or think of a useful improvement,
file an issue in this GitHub repository. Make sure to
include details about what you think should be changed
and why. If you're reporting a bug, please provide
a minimum working example so that a maintainer can
reproduce the bug.

### Modify the Source Code

If you know exactly what needs to change, you can also
submit a pull request to propose the change.

1. Fork it ( https://github.com/civisanalytics/civisml-extensions/fork ).
2. Install the dependencies (`pip install -r requirements.txt` and `pip install -r dev-requirements.txt`)
3. Make sure you are able to run the test suite locally (`pytest`)
4. Create a feature branch (`git checkout -b my-new-feature`)
5. Make your change. Don't forget tests
6. Make sure the test suite, including your new tests, passes (`pytest && flake8`)
7. Commit your changes (`git commit -am 'Add some feature'`)
8. Push to the branch (`git push origin my-new-feature`)
9. Create a new pull request
10. If the Travis build fails, address any issues

## Tips

- All pull requests must include test coverage. If you’re not sure how to test
  your changes, feel free to ask for help.
- Contributions must conform to PEP 8 and the NumPy/SciPy Documentation standards:
  - [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
  - [A Guide to NumPy/SciPy Documentation](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
- Code in this repo should also conform to the scikit-learn standards. See [their documentation](http://scikit-learn.org/stable/developers/index.html) for more information.
- Don’t forget to add your change to the [CHANGELOG](CHANGELOG.md). See
    [Keep a CHANGELOG](http://keepachangelog.com/) for guidelines.

Thank you for taking the time to contribute!
