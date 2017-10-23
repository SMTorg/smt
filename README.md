[![Build Status](https://travis-ci.org/SMTorg/smt.svg?branch=master)](https://travis-ci.org/SMTorg/smt) (Linux, OS X)

[![Build status](https://ci.appveyor.com/api/projects/status/o0303yw40sqqe88y?svg=true)](https://ci.appveyor.com/project/hwangjt/smt-52ku9) (Windows)

# README
This repository includes the codes for the surrogate model toolbox (SMT). SMT is a Python package and is distributed under the BSD license.

# Version
Version 0.1

# Required packages
Scipy    >= Version 0.15.1

Numpy    >= Version 1.9.2

Scikit-learn  >= Version 0.13.1

pyDOE >= Version 0.3.8

six

Cython

# Installation
Clone the repository from github then run:

```
pip install -e name_folder
```

# Tests
To run tests, first install the python testing framework using:

```
pip install git+https://github.com/OpenMDAO/testflo.git
```

and run

```
testflo
```

# Usage
For examples demonstrating how to use SMT, go to the 'examples' folder.

# Documentation
http://smt.readthedocs.io

# Contact
This repository was created by Mohamed Amine Bouhlel and is maintained by the MDOlab.
