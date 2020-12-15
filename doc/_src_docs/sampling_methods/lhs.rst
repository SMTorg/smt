Latin Hypercube sampling
========================

The LHS design is a statistical method for generating a quasi-random sampling distribution. It is among the most popular sampling techniques in computer experiments thanks to its simplicity and projection properties with high-dimensional problems. LHS is built as follows: we cut each dimension space, which represents a variable, into n
sections where n is the number of sampling points, and we put only one point in each section.

The LHS method uses the pyDOE package (Design of Experiments for Python) [1]_. Five criteria for the construction of LHS are implemented in SMT:

- Center the points within the sampling intervals.
- Maximize the minimum distance between points and place the point in a randomized location within its interval.
- Maximize the minimum distance between points and center the point within its interval.
- Minimize the maximum correlation coefficient.
- Optimize the design using the Enhanced Stochastic Evolutionary algorithm (ESE).

The four first criteria are the same than in pyDOE (for more details, see [1]_). The last criterion, ESE, is implemented by the authors of SMT (more details about such method could be found in [2]_).

.. [1] https://pythonhosted.org/pyDOE/index.html

.. [2] Jin, R. and Chen, W. and Sudjianto, A. (2005), "An efficient algorithm for constructing optimal design of computer experiments." Journal of Statistical Planning and Inference, 134:268-287.

Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.sampling_methods import LHS
  
  xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
  sampling = LHS(xlimits=xlimits)
  
  num = 50
  x = sampling(num)
  
  print(x.shape)
  
  plt.plot(x[:, 0], x[:, 1], "o")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()
  
::

  (50, 2)
  
.. figure:: lhs_Test_test_lhs.png
  :scale: 80 %
  :align: center

Options
-------

.. list-table:: List of options
  :header-rows: 1
  :widths: 15, 10, 20, 20, 30
  :stub-columns: 0

  *  -  Option
     -  Default
     -  Acceptable values
     -  Acceptable types
     -  Description
  *  -  xlimits
     -  None
     -  None
     -  ['ndarray']
     -  The interval of the domain in each dimension with shape nx x 2 (required)
  *  -  criterion
     -  c
     -  ['center', 'maximin', 'centermaximin', 'correlation', 'c', 'm', 'cm', 'corr', 'ese']
     -  ['str']
     -  criterion used to construct the LHS design c, m, cm and corr are abbreviation of center, maximin, centermaximin and correlation, respectively
  *  -  random_state
     -  None
     -  None
     -  ['NoneType', 'int', 'RandomState']
     -  Numpy RandomState object or seed number which controls random draws
