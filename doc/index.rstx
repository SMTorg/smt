.. SMT documentation master file, created by
   sphinx-quickstart on Sun Aug  6 19:36:14 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SMT: Surrogate Modeling Toolbox
-------------------------------

The surrogate modeling toolbox (SMT) is an open-source Python package consisting of libraries of surrogate modeling methods (e.g., radial basis functions, kriging), sampling methods, and benchmarking problems.
SMT is designed to make it easy for developers to implement new surrogate models in a well-tested and well-document platform, and for users to have a library of surrogate modeling methods with which to use and compare methods.

The code is available open-source on `GitHub <https://github.com/SMTorg/SMT>`_.

Cite us
-------
To cite SMT: M. A. Bouhlel and J. T. Hwang and N. Bartoli and R. Lafage and J. Morlier and J. R. R. A. Martins.  

`A Python surrogate modeling framework with derivatives. Advances in Engineering Software, 2019 <https://hal.archives-ouvertes.fr/hal-02294310/document>`_. 

.. code-block:: none

	@article{SMT2019,
		Author = {Mohamed Amine Bouhlel and John T. Hwang and Nathalie Bartoli and Rémi Lafage and Joseph Morlier and Joaquim R. R. A. Martins},
		Journal = {Advances in Engineering Software},
		Title = {A Python surrogate modeling framework with derivatives},
		pages = {102662},
		year = {2019},
		issn = {0965-9978},
		doi = {https://doi.org/10.1016/j.advengsoft.2019.03.005},
		Year = {2019}}


Focus on derivatives
--------------------

SMT is meant to be a general library for surrogate modeling (also known as metamodeling, interpolation, and regression), but its distinguishing characteristic is its focus on derivatives, e.g., to be used for gradient-based optimization.
A surrogate model can be represented mathematically as

.. math ::
  y = f(\mathbf{x}, \mathbf{xt}, \mathbf{yt}),

where
:math:`\mathbf{xt} \in \mathbb{R}^{nt \times nx}` contains the training inputs,
:math:`\mathbf{yt} \in \mathbb{R}^{nt}` contains the training outputs,
:math:`\mathbf{x} \in \mathbb{R}^{nx}` contains the prediction inputs,
and
:math:`y \in \mathbb{R}` contains the prediction outputs.
There are three types of derivatives of interest in SMT:

1. Derivatives (:math:`{dy}/{dx}`): derivatives of predicted outputs with respect to the inputs at which the model is evaluated.
2. Training derivatives (:math:`{dyt}/{dxt}`): derivatives of training outputs, given as part of the training data set, e.g., for gradient-enhanced kriging.
3. Output derivatives (:math:`{dy}/{dyt}`): derivatives of predicted outputs with respect to training outputs, representing how the prediction changes if the training outputs change and the surrogate model is re-trained.

Not all surrogate modeling methods support or are required to support all three types of derivatives; all are optional.

Documentation contents
----------------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   _src_docs/getting_started
   _src_docs/surrogate_models
   _src_docs/problems
   _src_docs/sampling_methods
   _src_docs/examples
   _src_docs/applications
   _src_docs/dev_docs


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
