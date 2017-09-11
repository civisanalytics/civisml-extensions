import os
from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as _in:
        return _in.read()


__version__ = None
exec(read('civismlext/_version.py'))

setup(version=__version__,
      name="civisml-extensions",
      author="Civis Analytics",
      author_email="opensource@civisanalytics.com",
      url="https://www.civisanalytics.com",
      description="scikit-learn-compatible estimators from Civis Analytics",
      packages=find_packages(),
      install_requires=read('requirements.txt').splitlines(),
      long_description=read('README.rst'),
      include_package_data=True,
      license="BSD-3")
