.. _Mixed Integer and Hierarchical Variables Types Specifications: 

Mixed Integer and Hierarchical usage (Variables, Sampling and Context)
======================================================================

Mixed integer variables types
-----------------------------

SMT provides the ``mixed_integer`` module to adapt existing surrogates to deal with categorical (or enumerate) and ordered integer variables using continuous relaxation.
For ordered variables, the values are rounded to the nearest values from a provided list. If, instead, only lower and upper bounds are provided, the list of all possible values will consists of the integers values between those bounds.

The user specifies x feature types through a list of types to be either:

- ``FLOAT``: a continuous feature,
- ``ORD``: an ordered valued feature,
- or a tuple ``(ENUM, n)`` where n is the number of levels of the catagorical feature (i.e. an enumerate with n values)

In the case of mixed integer sampling, bounds of each x feature have to be adapted to take into account feature types. While ``FLOAT`` and ``ORD`` feature still have an interval [lower bound, upper bound], the ``ENUM`` features bounds is defined by giving the enumeration/list of possible values (levels). 

For instance, if we have the following ``xtypes``: ``[FLOAT, ORD, (ENUM, 2), (ENUM, 3)]``, a compatible ``xlimits`` could be ``[[0., 4], [-10, 10], ["blue", "red"], ["short", "medium",  "long"]]``.

However, the functioning of ``ORD`` is twofold. As previously mentioned, it can be used like [lower bound, upper bound], in this case [0,5] will corresponds to [0,1,2,3,4,5]. But, on the other hand, ``ORD`` can be used as an enumeration/list of possible values (levels), in this case ["0","5","6"] will corresponds to [0,5,6]. However, these ordered values should be string representation of integer. Details can be found in [1]_ .

Hierarchical variables roles
----------------------------

The ``mixed_integer`` module uses the framework of Audet et al. [2]_ to manage both mixed variables and hierarchical variables. We distinguish dimensional (or meta) variables which are a special type of variables that may affect the dimension of the problem and decide if some other decreed variables are included or excluded. The variable size problem can also includes neutral variables that are always included and actives. 

The user specifies x feature role through a list of roles amongst:

- ``META``: a dimensional feature,
- ``DECREED``: an ordered or continuous decreed feature: either included or excluded in the variable-size problem,
- ``NEUTRAL``: a neutral feature, part of the fixed-size problem

Note that we do not consider decreed categorical variable. The decreed variables are always continuous or ordered.

Mixed and hierarchical specifications
-------------------------------------

The ``XSpecs`` class helps implements the types, limits and roles of each variables as follows.

  .. autoclass:: smt.utils.kriging.XSpecs

Mixed integer sampling method
-----------------------------

In the case of mixed integer sampling, bounds of each x feature have to be adapted to take into account feature types. While ``FLOAT`` and ``ORD`` feature still have an interval [lower bound, upper bound], the ``ENUM`` features bounds is defined by giving the enumeration/list of possible values (levels). 

For instance, if we have the following ``xtypes``: ``[FLOAT, ORD, (ENUM, 2), (ENUM, 3)]``, a compatible ``xlimits`` could be ``[[0., 4], [-10, 10], ["blue", "red"], ["short", "medium",  "long"]]``.

However, the functioning of ``ORD`` is twofold. As previously mentioned, it can be used like [lower bound, upper bound], in this case [0,5] will corresponds to [0,1,2,3,4,5]. But, on the other hand, ``ORD`` can be used as an enumeration/list of possible values (levels), in this case ["0","5","6"] will corresponds to [0,5,6].

To use a sampling method with mixed integer typed features, the user instanciates a ``MixedIntegerSamplingMethod`` with a given sampling method.
The ``MixedIntegerSamplingMethod`` implements the ``SamplingMethod`` interface and decorates the original sampling method to provide a DOE while conforming to integer and categorical types.

Example of mixed integer LHS sampling method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib import colors
  
  from smt.sampling_methods import LHS
  from smt.surrogate_models import XType, XSpecs
  from smt.applications.mixed_integer import MixedIntegerSamplingMethod
  
  xtypes = [XType.FLOAT, (XType.ENUM, 2)]
  xlimits = [[0.0, 4.0], ["blue", "red"]]
  xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
  
  sampling = MixedIntegerSamplingMethod(LHS, xspecs, criterion="ese")
  
  num = 40
  x = sampling(num)
  
  cmap = colors.ListedColormap(xlimits[1])
  plt.scatter(x[:, 0], np.zeros(num), c=x[:, 1], cmap=cmap)
  plt.show()
  
.. figure:: Mixed_Hier_usage_TestMixedInteger_run_mixed_integer_lhs_example.png
  :scale: 80 %
  :align: center

Mixed integer context
---------------------

The ``MixedIntegerContext`` class helps the user to use mixed integer sampling methods and surrogate models consistently by acting as a factory for those objects given a x specification: (xtypes, xlimits). 

  .. autoclass:: smt.applications.mixed_integer.MixedIntegerContext

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.__init__

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.build_sampling_method

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.build_surrogate_model

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.cast_to_discrete_values

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.fold_with_enum_index

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.unfold_with_enum_mask

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.cast_to_mixed_integer

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.cast_to_enum_value

Example of mixed integer context usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib import colors
  from mpl_toolkits.mplot3d import Axes3D
  
  from smt.sampling_methods import LHS, Random
  from smt.surrogate_models import KRG, XType, XSpecs
  from smt.applications.mixed_integer import MixedIntegerContext
  
  xtypes = [XType.ORD, XType.FLOAT, (XType.ENUM, 4)]
  xlimits = [[0, 5], [0.0, 4.0], ["blue", "red", "green", "yellow"]]
  xspecs = XSpecs(xtypes=xtypes, xlimits=xlimits)
  
  def ftest(x):
      return (x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]) * (x[:, 2] + 1)
  
  # context to create consistent DOEs and surrogate
  mixint = MixedIntegerContext(xspecs=xspecs)
  
  # DOE for training
  lhs = mixint.build_sampling_method(LHS, criterion="ese")
  
  num = mixint.get_unfolded_dimension() * 5
  print("DOE point nb = {}".format(num))
  xt = lhs(num)
  yt = ftest(xt)
  
  # Surrogate
  sm = mixint.build_kriging_model(KRG())
  sm.set_training_values(xt, yt)
  sm.train()
  
  # DOE for validation
  rand = mixint.build_sampling_method(Random)
  xv = rand(50)
  yv = ftest(xv)
  yp = sm.predict_values(xv)
  
  plt.plot(yv, yv)
  plt.plot(yv, yp, "o")
  plt.xlabel("actual")
  plt.ylabel("prediction")
  
  plt.show()
  
::

  DOE point nb = 15
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 50
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: Mixed_Hier_usage_TestMixedInteger_run_mixed_integer_context_example.png
  :scale: 80 %
  :align: center

References
----------

.. [1] Saves, P. and Diouane, Y. and Bartoli, N. and Lefebvre, T. and Morlier, J. (2022). A general square exponential kernel to handle mixed-categorical variables for Gaussian process. AIAA Aviation 2022 Forum. 

.. [2] Audet, C., Hallé-Hannan, E. and Le Digabel, S. A General Mathematical Framework for Constrained Mixed-variable Blackbox Optimization Problems with Meta and Categorical Variables. Oper. Res. Forum 4, 12 (2023). 
