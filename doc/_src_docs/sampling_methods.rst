Sampling methods
================

SMT contains a library of sampling methods used to generate sets of points in the input space,
either for training or for prediction.
These are listed below.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   sampling_methods/random
   sampling_methods/lhs
   sampling_methods/full_factorial

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.sampling_methods import Random
  
  xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
  sampling = Random(xlimits=xlimits)
  
  num = 50
  x = sampling(num)
  
  print(x.shape)
  
  plt.plot(x[:, 0], x[:, 1], "o")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()
  
::

  (50, 2)
  
.. figure:: sampling_methods_Test_test_random.png
  :scale: 80 %
  :align: center

Problem class API
-----------------

.. autoclass:: smt.sampling_methods.sampling_method.SamplingMethod

  .. automethod:: smt.sampling_methods.sampling_method.SamplingMethod.__init__

  .. automethod:: smt.sampling_methods.sampling_method.SamplingMethod.__call__
