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


Mixed-Integer Surrogate with Gower Distance
===========================================

Another implemented method is using a basic mixed integer kernel based on the Gower distance between two points.
When constructing the correlation kernel, the distance is redefined as :math:`\Delta= \Delta_{cont} + \Delta_{cat}`, with :math:`\Delta_{cont}` the continuous distance as usual and :math:`\Delta_ {cat}` the categorical distance defined as the number of categorical variables that differs from one point to another.

For example, the Gower Distance between ``[1,'red', 'medium']`` and ``[1.2,'red', 'large']`` is :math:`\Delta= 0.2+ (0` ``'red'`` :math:`=` ``'red'`` :math:`+ 1` ``'medium'`` :math:`\neq` ``'large'``  ) :math:`=1.2`

Example of mixed-integer Gower Distance model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  from smt.applications.mixed_integer import MixedIntegerSurrogateModel, ENUM
  from smt.surrogate_models import KRG
  import matplotlib.pyplot as plt
  import numpy as np
  
  xt = np.linspace(1.0, 5.0, 5)
  x_train = np.array(["%.2f" % i for i in xt], dtype=object)
  yt = np.array([0.0, 1.0, 1.5, 0.5, 1.0])
  
  xlimits = [["0.0", "1.0", " 2.0", "3.0", "4.0"]]
  
  # Surrogate
  sm = MixedIntegerSurrogateModel(
      use_gower_distance=True,
      xtypes=[(ENUM, 5)],
      xlimits=xlimits,
      surrogate=KRG(theta0=[1e-2]),
  )
  sm.set_training_values(x_train, yt)
  sm.train()
  
  # DOE for validation
  num = 101
  x = np.linspace(0, 5, num)
  x_pred = np.array(["%.2f" % i for i in x], dtype=object)
  y = sm.predict_values(x_pred)
  
  plt.plot(xt, yt, "o")
  plt.plot(x, y)
  plt.xlabel("actual")
  plt.ylabel("prediction")
  plt.show()
  
::

  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 101
     
     Predicting ...
     Predicting - done. Time (sec):  0.0039895
     
     Prediction time/pt. (sec) :  0.0000395
     
  
.. figure:: mixed_integer_TestMixedInteger_test_mixed_gower.png
  :scale: 80	 %
  :align: center


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
  
  cmap = colors.ListedColormap(xlimits[1])
  plt.scatter(x[:, 0], np.zeros(num), c=x[:, 1], cmap=cmap)
  plt.show()
  
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
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 50
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: mixed_integer_TestMixedInteger_run_mixed_integer_context_example.png
  :scale: 80 %
  :align: center



