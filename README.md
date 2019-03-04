[![Build Status](https://travis-ci.org/SMTorg/smt.svg?branch=master)](https://travis-ci.org/SMTorg/smt) [![Build status](https://ci.appveyor.com/api/projects/status/o0303yw40sqqe88y?svg=true)](https://ci.appveyor.com/project/hwangjt/smt-52ku9) [![Coverage Status](https://coveralls.io/repos/github/SMTorg/smt/badge.svg)](https://coveralls.io/github/SMTorg/smt)

# Surrogate Modeling Toolbox
The surrogate modeling toolbox (SMT) is a Python package that contains a collection of surrogate modeling methods, sampling techniques, and benchmarking functions. This package provides a library of surrogate models that is simple to use and facilitates the implementation of additional methods.
SMT is different from existing surrogate modeling libraries because of its emphasis on derivatives, including training derivatives used for gradient-enhanced modeling, prediction derivatives, and derivatives with respect to the training data.
It also includes new surrogate models that are not available elsewhere: kriging by partial-least squares reduction and energy-minimizing spline interpolation.
SMT is documented using custom tools for embedding automatically-tested code and dynamically-generated plots to produce high-quality user guides with minimal effort from contributors.
SMT is distributed under the New BSD license.

# Version
Version 0.2

# Required packages
SMT depends on the following modules: numpy, scipy, sk-learn, pyDOE2 and Cython. 

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
For examples demonstrating how to use SMT, go to the 'smt/examples' folder.

# Documentation
http://smt.readthedocs.io

# Contact
This repository was created by Mohamed Amine Bouhlel and is maintained by the [MDOlab](https://github.com/mdolab).


Email: mbouhlel@umich.edu or bouhlel.mohamed.amine@gmail.com
