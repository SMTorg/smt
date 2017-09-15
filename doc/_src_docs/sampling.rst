Sampling methods
================

SMT contains a library of sampling methods used to generate sets of points in the input space,
either for training or for prediction.
These are listed below.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   sampling/random
   sampling/lhs
   sampling/full_factorial
   sampling/clustered

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.sampling import Random
  
  xlimits = np.array([
      [0., 4.],
      [0., 3.],
  ])
  sampling = Random(xlimits=xlimits)
  
  num = 50
  x = sampling(num)
  
  print(x.shape)
  
  plt.plot(x[:, 0], x[:, 1], 'o')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.show()
  
::

  (50, 2)
  
.. figure:: sampling_Test_test_random.png
  :scale: 80 %
  :align: center

Problem class API
-----------------

.. autoclass:: smt.sampling.sampling.Sampling

  .. automethod:: smt.sampling.sampling.Sampling.__init__

  .. automethod:: smt.sampling.sampling.Sampling.__call__
