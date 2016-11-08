# README
This repository contains the code for The Surrogate Models Toolbox (SMT).

# Version
Version 0.0.1

# How do I use the SMT?
Clone the repository from github.
Go to the main folder SMT.
Install the toolbox by running:
python setup.py install

# How do I test the SMT?
## Linux
Go to the folder examples and run:
sudo python test.py

## Windows
The Inverse Distance Weighting (IDW) model is not available with windows.
Open the "test.py" file with an editor and comment line 6 and from line
64 through line 71. Then, run:
python test.py

This repository was created by Mohamed Amine Bouhlel and is maintained by the MDO Lab
