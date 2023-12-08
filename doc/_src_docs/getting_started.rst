Getting started
===============

Prerequisites
-------------

SMT setup requires :code:`numpy`, ensure you have it installed.

As some surrogates are written in C++, SMT setup uses :code:`Cython` to compile them.
If :code:`Cython` is not found, SMT setup tries to install it, but as this step is not robust for all
environments it is better to ensure you have it properly installed before hand.
If compilation with :code:`Cython` fails then RBF, IDW, RMTB and RMTC surrogates will not be available.

Note: if you use Anaconda Python distribution, it is best to install 
dependencies: :code:`numpy`, :code:`scipy`, :code:`sklearn`, :code:`Cython` 
with :code:`conda` command then use :code:`pip` command as follows to terminate the SMT installation.

Installing
----------

To install the `latest released version <https://pypi.org/project/smt/>`_ of SMT: 

:code:`pip install smt`

Or else to install the current version from the GitHub repository master:

:code:`pip install git+https://github.com/SMTOrg/smt.git@master`

If you want to contribute to SMT, see :ref:`contributing` section.

Notebooks
---------

Several notebooks are available to get up to speed with SMT:

* `General <https://github.com/SMTorg/smt/blob/master/tutorial/SMT_Tutorial.ipynb>`_
* `Handling of Noise <https://github.com/SMTorg/smt/blob/master/tutorial/SMT_Noise.ipynb>`_
* `Handling of Mixed Integer variables <https://github.com/SMTorg/smt/blob/master/tutorial/SMT_MixedInteger_application.ipynb>`_
* `Efficient Global Optimization application <https://github.com/SMTorg/smt/blob/master/tutorial/SMT_EGO_application.ipynb>`_

Uninstalling
------------

If you want to uninstall SMT:

:code:`pip uninstall smt`
