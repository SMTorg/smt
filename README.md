[![Build Status](https://travis-ci.org/SMTorganization/SMT.svg?branch=master)](https://travis-ci.org/SMTorganization/SMT) (Linux)

[![Build status](https://ci.appveyor.com/api/projects/status/o0303yw40sqqe88y?svg=true)](https://ci.appveyor.com/project/hwangjt/smt-52ku9) (Windows)

# README
This repository includes the codes for the surrogate model toolbox (SMT).

# Version
Version beta

# Required packages
Scipy    >= Version 0.15.1

Numpy    >= Version 1.9.2

Sklearn  >= Version 0.13.1

pyDOE >= Version 0.3.8

# Installation
Clone the repository from github then run:

```
pip install -e name_folder
```

or

Go to the main folder.

Install the toolbox by running:

```
sudo python setup.py install        (Linux)
```

or

```
python setup.py install             (Windows)
```

# Usage

For examples demonstrating how to use SMT, go to the 'examples' folder.

# Tests

To run tests, first install the python testing framework using:

```
pip install git+https://github.com/OpenMDAO/testflo.git
```

and run

```
testflo
```

# Contact
This repository was created by Mohamed Amine Bouhlel and is maintained by the MDOlab.
