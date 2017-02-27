Linux: [![Build Status](https://travis-ci.org/hwangjt/SMT.svg?branch=master)](https://travis-ci.org/hwangjt/SMT)   
OS X: [![Build status](https://ci.appveyor.com/api/projects/status/1dd3wovs981r86e0?svg=true)](https://ci.appveyor.com/project/hwangjt/smt)
# README
This repository includes the codes for the surrogate model toolbox (SMT).

# Version
Version beta

# Required packages
Scipy    >= Version 0.15.1

Numpy    >= Version 1.9.2

Sklearn  >= Version 0.13.1

pyDOE >= Version 0.3.8

# How do I use the SMT?
Clone the repository from github then run:

pip install -e name_folder

or

Go to the main folder.

Install the toolbox by running:

sudo python setup.py install        (Linux)

or

python setup.py install             (Windows)

# How do I test the SMT?

Go to the folder 'examples' and run:

## Linux
python test_linux.py

If several versions of Python are installed on the computer, run:

sudo python test_linux.py

## Windows
python test_windows.py

# Contact
This repository was created by Mohamed Amine Bouhlel and is maintained by the MDOlab.
