# README
This repository includes the codes for the surrogate model toolbox (SMT).

# Version
Version beta

# Required packages
Scipy    >= Version 0.15.1
Numpy    >= Version 1.9.2
Sklearn  >= Version 0.13.1

# How do I use the SMT?
Clone the repository from github.

Go to the main folder 'SMT'.

Install the toolbox by running:

sudo python setup.py install        (Linux)

or 

python setup.py install             (Windows)

# How do I test the SMT?
## Linux
Go to the folder 'examples' and run:

python test.py

If several versions of Python are installed on the computer, run: 

sudo python test.py

## Windows
The inverse distance weighting (IDW) model is not available with Windows.

Open the "test.py" file with an editor and comment line 6 and from line
64 through line 71. Then, run:

python test.py

# Contact
This repository was created by Mohamed Amine Bouhlel and is maintained by the MDOlab.
