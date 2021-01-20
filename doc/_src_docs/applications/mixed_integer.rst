.. _Mixed-Integer Sampling and Surrogate:

Mixed-Integer Sampling and Surrogate (Continuous Relaxation)
============================================================

SMT provides the ``mixed_integer`` module to adapt existing surrogates to deal with
categorical (or enumerate) and integer variables using continuous relaxation.

For integer variables, values are rounded to the closer integer.
For enum variables, as many x features as enumerated levels are created with [0, 1] bounds 
and the max of these feature float values will correspond to the choice of one the enum value. 

For instance, for a categorical variable (one feature of x) with three levels ["blue", "red", "green"],
3 continuous float features x0, x1, x2 are created, the max(x0, x1, x2), 
let say x1, will give "red" as the value for the original categorical feature. 

The user specifies x feature types through a list of types to be either:

- ``FLOAT``: a continuous feature,
- ``INT``: an integer valued feature,
- or a tuple ``(ENUM, n)`` where n is the number of levels of the catagorical feature (i.e. an enumerate with n values)

In the case of mixed integer sampling, bounds of each x feature have to be adapted 
to take into account feature types. While FLOAT and INT feature still have an interval
[lower bound, upper bound], the ENUM features bounds is defined by giving the enumeration/list
of possible values (levels). 

For instance, if we have the following ``xtypes``: ``[FLOAT, INT, (ENUM, 2), (ENUM, 3)]``, 
a compatible ``xlimits`` could be ``[[0., 4], [-10, 10], ["blue", "red"], ["short", "medium", "long"]]``

Mixed integer sampling method
-----------------------------

To use a sampling method with mixed integer typed features, the user instanciates
a ``MixedIntegerSamplingMethod`` with a given sampling method.
The ``MixedIntegerSamplingMethod`` implements the ``SamplingMethod`` interface 
and decorates the original sampling method to provide a DOE while conforming to integer 
and categorical types.

Example of mixed-integer LHS sampling method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib import colors
  
  from smt.sampling_methods import LHS
  from smt.applications.mixed_integer import (
      FLOAT,
      INT,
      ENUM,
      MixedIntegerSamplingMethod,
  )
  
  xtypes = [FLOAT, (ENUM, 2)]
  xlimits = [[0.0, 4.0], ["blue", "red"]]
  sampling = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="ese")
  
  num = 40
  x = sampling(num)
  
  print(x.shape)
  
  cmap = colors.ListedColormap(xlimits[1])
  plt.scatter(x[:, 0], np.zeros(num), c=x[:, 1], cmap=cmap)
  plt.show()
  
::

  (40, 2)
  
.. figure:: mixed_integer_TestMixedInteger_run_mixed_integer_lhs_example.png
  :scale: 80 %
  :align: center

Mixed integer surrogate
-----------------------

To use a surrogate with mixed integer constraints, the user instanciates
a ``MixedIntegerSurrogateModel`` with the given surrogate.
The ``MixedIntegerSurrogateModel`` implements the ``SurrogateModel`` interface 
and decorates the given surrogate while respecting integer and categorical types.

Example of mixed-integer Polynomial (QP) surrogate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.surrogate_models import QP
  from smt.applications.mixed_integer import MixedIntegerSurrogateModel, INT
  
  xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
  yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])
  
  # xtypes = [FLOAT, INT, (ENUM, 3), (ENUM, 2)]
  # FLOAT means x1 continuous
  # INT means x2 integer
  # (ENUM, 3) means x3, x4 & x5 are 3 levels of the same categorical variable
  # (ENUM, 2) means x6 & x7 are 2 levels of the same categorical variable
  
  sm = MixedIntegerSurrogateModel(xtypes=[INT], xlimits=[[0, 4]], surrogate=QP())
  sm.set_training_values(xt, yt)
  sm.train()
  
  num = 100
  x = np.linspace(0.0, 4.0, num)
  y = sm.predict_values(x)
  
  plt.plot(xt, yt, "o")
  plt.plot(x, y)
  plt.xlabel("x")
  plt.ylabel("y")
  plt.legend(["Training data", "Prediction"])
  plt.show()
  
::

  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: mixed_integer_TestMixedInteger_run_mixed_integer_qp_example.png
  :scale: 80 %
  :align: center

Mixed integer context
---------------------

the ``MixedIntegerContext`` class helps the user to use mixed integer sampling methods and surrogate models consistently 
by acting as a factory for those objects given a x specification: (xtypes, xlimits). 

.. autoclass:: smt.applications.mixed_integer.MixedIntegerContext

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.__init__

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.build_sampling_method

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.build_surrogate_model

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.cast_to_discrete_values

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.fold_with_enum_index

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.unfold_with_enum_mask

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.cast_to_mixed_integer

  .. automethod:: smt.applications.mixed_integer.MixedIntegerContext.cast_to_enum_value

Example of mixed-integer context usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  from matplotlib import colors
  from mpl_toolkits.mplot3d import Axes3D
  
  from smt.surrogate_models import KRG
  from smt.sampling_methods import LHS, Random
  from smt.applications.mixed_integer import MixedIntegerContext, FLOAT, INT, ENUM
  
  xtypes = [INT, FLOAT, (ENUM, 4)]
  xlimits = [[0, 5], [0.0, 4.0], ["blue", "red", "green", "yellow"]]
  
  def ftest(x):
      return (x[:, 0] * x[:, 0] + x[:, 1] * x[:, 1]) * (x[:, 2] + 1)
  
  # context to create consistent DOEs and surrogate
  mixint = MixedIntegerContext(xtypes, xlimits)
  
  # DOE for training
  lhs = mixint.build_sampling_method(LHS, criterion="ese")
  
  num = mixint.get_unfolded_dimension() * 5
  print("DOE point nb = {}".format(num))
  xt = lhs(num)
  yt = ftest(xt)
  
  # Surrogate
  sm = mixint.build_surrogate_model(KRG())
  print(xt)
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

  DOE point nb = 30
  [[4.         1.70521642 0.        ]
   [4.         2.33193344 1.        ]
   [5.         3.94276469 0.        ]
   [2.         2.18545678 3.        ]
   [3.         0.55708744 1.        ]
   [4.         3.47963595 2.        ]
   [1.         3.1637495  0.        ]
   [2.         0.68953597 3.        ]
   [3.         3.84977525 1.        ]
   [4.         0.09721383 0.        ]
   [1.         2.13247484 2.        ]
   [5.         1.28923662 3.        ]
   [1.         3.61081518 0.        ]
   [4.         1.93331791 2.        ]
   [0.         3.27935675 1.        ]
   [2.         2.47092774 0.        ]
   [3.         1.81639775 0.        ]
   [2.         3.40344424 3.        ]
   [2.         1.34244476 2.        ]
   [2.         3.023775   2.        ]
   [5.         2.53788277 3.        ]
   [0.         0.45198974 2.        ]
   [1.         1.16536925 1.        ]
   [1.         0.31822681 1.        ]
   [0.         0.98230261 0.        ]
   [1.         1.54645493 1.        ]
   [4.         0.87159838 2.        ]
   [3.         2.70215936 2.        ]
   [3.         2.83393431 1.        ]
   [3.         0.20180401 3.        ]]
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 50
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: mixed_integer_TestMixedInteger_run_mixed_integer_context_example.png
  :scale: 80 %
  :align: center



