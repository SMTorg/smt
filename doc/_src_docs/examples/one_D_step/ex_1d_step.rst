1-D step-like data set
======================

.. code-block:: python

  import numpy as np
  
  
  def get_one_d_step():
      xt = np.array([
          0.0000,    0.4000,    0.6000,    0.7000,    0.7500,
          0.7750,    0.8000,    0.8500,    0.8750,    0.9000,
          0.9250,    0.9500,    0.9750,    1.0000,    1.0250,
          1.0500,    1.1000,    1.2000,    1.3000,    1.4000,
          1.6000,    1.8000,    2.0000,
      ], dtype=np.float64)
      yt = np.array([
          0.0130,     0.0130,     0.0130,     0.0130,   0.0130,
          0.0130,     0.0130,     0.0132,     0.0135,   0.0140,
          0.0162,     0.0230,     0.0275,     0.0310,   0.0344,
          0.0366,     0.0396,     0.0410,     0.0403,   0.0390,
          0.0360,     0.0350,     0.0345,
      ], dtype=np.float64)
  
      xlimits = np.array([[0.0, 2.0]])
  
      return xt, yt, xlimits
  
  
  def plot_one_d_step(xt, yt, limits, interp):
      import numpy as np
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
  
      num = 500
      x = np.linspace(0., 2., num)
      y = interp.predict_values(x)[:, 0]
  
      plt.plot(x, y)
      plt.plot(xt, yt, 'o')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.show()
  

RMTB
----

.. code-block:: python

  from smt.surrogate_models import RMTB
  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  
  xt, yt, xlimits = get_one_d_step()
  
  interp = RMTB(num_ctrl_pts=100, xlimits=xlimits, nonlinear_maxiter=20,
      solver_tolerance=1e-16, energy_weight=1e-14, regularization_weight=0.)
  interp.set_training_values(xt, yt)
  interp.train()
  
  plot_one_d_step(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTB
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 23
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0000000
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0149999
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.0149999
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.801195655e-08 2.334012037e-13
              Solving for output 0 - done. Time (sec):  0.0000000
           Solving initial startup problem (n=100) - done. Time (sec):  0.0000000
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.453373643e-11 2.297930422e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.284303425e-11 2.269770253e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.589638783e-10 1.407667295e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.403626670e-10 9.942133065e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.032120050e-10 2.775019485e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.993674882e-11 1.196201530e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 8.283649219e-12 9.067876728e-15
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.030161663e-12 8.531007288e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 3.563548349e-13 8.459283480e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 3.204644635e-13 8.458993607e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.583093587e-13 8.457968365e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 5.345624535e-14 8.454274624e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 4.315804707e-14 8.454042909e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.198658550e-14 8.453479025e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.630990861e-15 8.453310845e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.000068980e-15 8.453272915e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 8.208460651e-16 8.453272829e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 6.840939871e-16 8.453272397e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 5.111136355e-16 8.453271921e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 5.140623998e-16 8.453271388e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.750462375e-16 8.453270632e-15
              Solving for output 0 - done. Time (sec):  0.1250000
           Solving nonlinear problem (n=100) - done. Time (sec):  0.1250000
        Solving for degrees of freedom - done. Time (sec):  0.1250000
     Training - done. Time (sec):  0.1399999
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.surrogate_models import RMTC
  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  
  xt, yt, xlimits = get_one_d_step()
  
  interp = RMTC(num_elements=40, xlimits=xlimits, nonlinear_maxiter=20,
      solver_tolerance=1e-16, energy_weight=1e-14, regularization_weight=0.)
  interp.set_training_values(xt, yt)
  interp.train()
  
  plot_one_d_step(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTC
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 23
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0000000
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0000000
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.0000000
        Solving for degrees of freedom ...
           Solving initial startup problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.470849329e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.312755056e-12 2.493686438e-14
              Solving for output 0 - done. Time (sec):  0.0150001
           Solving initial startup problem (n=82) - done. Time (sec):  0.0150001
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.484146500e-12 2.493686322e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032461760e-12 2.483319861e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.685193693e-11 2.389447567e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 6.759024204e-11 1.951447473e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 4.606277256e-11 1.520729527e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 9.660287740e-12 1.149722383e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.350804583e-12 1.128828757e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 8.168719836e-13 1.110474728e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.688728813e-13 1.109149617e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.489722108e-13 1.109109613e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.466495915e-13 1.109082624e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 3.182788595e-14 1.108960411e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.727238093e-14 1.108947747e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 3.952132332e-15 1.108940943e-14
                 Iteration (num., iy, grad. norm, func.) :  13   0 5.579782218e-16 1.108940356e-14
                 Iteration (num., iy, grad. norm, func.) :  14   0 2.221817888e-17 1.108940340e-14
              Solving for output 0 - done. Time (sec):  0.0939999
           Solving nonlinear problem (n=82) - done. Time (sec):  0.0939999
        Solving for degrees of freedom - done. Time (sec):  0.1090000
     Training - done. Time (sec):  0.1090000
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center
