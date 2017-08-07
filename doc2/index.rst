.. SMT documentation master file, created by
   sphinx-quickstart on Sun Aug  6 19:36:14 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SMT: Surrogate Modeling Toolbox
===============================

The surrogate model toolbox (SMT) is an open-source Python package consisting of libraries of surrogate modeling methods (e.g., radial basis functions, kriging), sampling methods, and benchmarking problems.
SMT is designed to make it easy for developers to implement new surrogate models in a well-tested and well-document platform, and for users to have a library of surrogate modeling methods with which to use and compare methods.

The code is available open-source on `GitHub <http://www.github.org/SMTorg/SMT/>`_.

Focus on derivatives
--------------------

SMT is meant to be a general library for surrogate modeling (also known as metamodeling, interpolation, and regression), but its distinguishing characteristic is its focus on derivatives, e.g., to be used for gradient-based optimization.
A surrogate model can be represented mathematically as

.. math ::

  y = f(x, xt, yt) ,

where :math:`xt` contains the training inputs, :math:`yt` contains the training outputs, :math:`x` contains the prediction inputs, and :math:`y` contains the prediction outputs.
There are three types of derivatives of interest in SMT:

1. Derivatives (:math:`{dy}/{dx}`): derivatives of predicted outputs with respect to the inputs at which the model is evaluated.
2. Training derivatives (:math:`{dyt}/{dxt}`): derivatives of training outputs, given as part of the training data set, e.g., for gradient-enhanced kriging.
3. Output derivatives (:math:`{dy}/{dyt}`): derivatives of predicted outputs with respect to training outputs, representing how the prediction changes if the training outputs change and the surrogate model is re-trained.

Not all surrogate modeling methods support or are required to support all three types of derivatives; all are optional.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2

   _src_docs/methods
   _src_docs/problems


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
