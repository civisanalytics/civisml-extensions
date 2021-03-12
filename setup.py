import os
from setuptools import find_packages, setup

CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
]


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as _in:
        return _in.read()


_VERSION = '0.3.1'

setup(version=_VERSION,
      name="civisml-extensions",
      author="Civis Analytics",
      author_email="opensource@civisanalytics.com",
      url="https://www.civisanalytics.com",
      description="scikit-learn-compatible estimators from Civis Analytics",
      packages=find_packages(),
      install_requires=read('requirements.txt').splitlines(),
      long_description=read('README.rst'),
      long_description_content_type='text/x-rst',
      include_package_data=True,
      license="BSD-3",
      classifiers=CLASSIFIERS,
      python_requires=">=3.6")
