Surrogate modeling methods
==========================

SMT contains the surrogate modeling methods listed below.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   methods/rbf
   methods/idw
   methods/rmts
   methods/ls
   methods/qp
   methods/krg
   methods/kpls
   methods/kplsk
   methods/gekpls


Usage
-----

.. code-block:: python

  import numpy as np
  import matplotlib.pyplot as plt
  
  from smt.methods import RBF
  
  xt = np.array([0., 1., 2., 3., 4.])
  yt = np.array([0., 1., 1.5, 0.5, 1.0])
  
  sm = RBF(d0=5)
  sm.set_training_values(xt, yt)
  sm.train()
  
  num = 100
  x = np.linspace(0., 4., num)
  y = sm.predict_values(x)
  
  plt.plot(xt, yt, 'o')
  plt.plot(x, y)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend(['Training data', 'Prediction'])
  plt.show()
  
::

  ___________________________________________________________________________
     
                                      RBF
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 5
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Initializing linear solver ...
           Performing LU fact. (5 x 5 mtx) ...
           Performing LU fact. (5 x 5 mtx) - done. Time (sec):  0.0006263
        Initializing linear solver - done. Time (sec):  0.0006621
        Solving linear system (col. 0) ...
           Back solving (5 x 5 mtx) ...
           Back solving (5 x 5 mtx) - done. Time (sec):  0.0003493
        Solving linear system (col. 0) - done. Time (sec):  0.0003829
     Training - done. Time (sec):  0.0015471
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000391
     
     Prediction time/pt. (sec) :  0.0000004
     
  
.. figure:: methods_Test_test_rbf.png
  :scale: 80 %
  :align: center

SM class API
------------

All surrogate modeling methods implement the following API, though some of the functions in the API are not supported by all methods.

.. autoclass:: smt.methods.sm.SM

  .. automethod:: smt.methods.sm.SM.__init__

  .. automethod:: smt.methods.sm.SM.set_training_values

  .. automethod:: smt.methods.sm.SM.set_training_derivatives

  .. automethod:: smt.methods.sm.SM.train

  .. automethod:: smt.methods.sm.SM.predict_values

  .. automethod:: smt.methods.sm.SM.predict_derivatives

  .. automethod:: smt.methods.sm.SM.predict_output_derivatives

  .. automethod:: smt.methods.sm.SM.predict_variances
