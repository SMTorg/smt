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

  from smt.methods import RMTB
  from smt.examples.one_d_step import get_one_d_step, plot_one_d_step
  
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
           Computing dof2coeff - done. Time (sec):  0.0000031
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003862
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0011950
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0004072
        Pre-computing matrices - done. Time (sec):  0.0020399
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 6.450151056e-08 2.293337503e-13
              Solving for output 0 - done. Time (sec):  0.0121858
           Solving initial startup problem (n=100) - done. Time (sec):  0.0122478
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.427045627e-11 2.251640048e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.256597888e-11 2.223872768e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.514422134e-10 1.379388089e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.222722546e-10 9.339026419e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.479626694e-11 2.439102876e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.740645412e-11 1.129607061e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.342246679e-11 9.638182798e-15
                 Iteration (num., iy, grad. norm, func.) :   6   0 3.720222983e-12 8.658769707e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 9.922370793e-13 8.487482447e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 2.850573039e-13 8.458934917e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 8.579820827e-14 8.455174441e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 2.487322446e-14 8.453471116e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 6.665241145e-15 8.453343840e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 4.248180503e-15 8.453288884e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.628906498e-15 8.453276326e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.530815036e-15 8.453276311e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 2.145637319e-15 8.453275212e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.195135530e-16 8.453272655e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.862073211e-16 8.453271480e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.785965429e-16 8.453271445e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 8.730866180e-17 8.453270568e-15
              Solving for output 0 - done. Time (sec):  0.2336931
           Solving nonlinear problem (n=100) - done. Time (sec):  0.2337532
        Solving for degrees of freedom - done. Time (sec):  0.2460611
     Training - done. Time (sec):  0.2484272
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007849
     
     Prediction time/pt. (sec) :  0.0000016
     
  
.. figure:: ex_1d_step_Test_test_rmtb.png
  :scale: 80 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.methods import RMTC
  from smt.examples.one_d_step import get_one_d_step, plot_one_d_step
  
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
           Computing dof2coeff - done. Time (sec):  0.0009041
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003519
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0013590
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0005310
        Pre-computing matrices - done. Time (sec):  0.0032158
        Solving for degrees of freedom ...
           Solving initial startup problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.470849329e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.247786014e-10 2.493640129e-14
              Solving for output 0 - done. Time (sec):  0.0071559
           Solving initial startup problem (n=82) - done. Time (sec):  0.0072250
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.484040934e-12 2.493593876e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032474820e-12 2.483229621e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.666678164e-11 2.386808376e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 4.541394429e-11 1.692063537e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.217128926e-11 1.163306229e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 6.384563217e-12 1.128973044e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.534184875e-12 1.111315209e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 3.783034725e-13 1.109342182e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 8.107378007e-14 1.109013567e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.971003371e-14 1.108949679e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 4.110418974e-15 1.108941032e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 4.965794809e-16 1.108940357e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 2.081326043e-17 1.108940340e-14
              Solving for output 0 - done. Time (sec):  0.0861020
           Solving nonlinear problem (n=82) - done. Time (sec):  0.0861621
        Solving for degrees of freedom - done. Time (sec):  0.0934823
     Training - done. Time (sec):  0.0971231
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005312
     
     Prediction time/pt. (sec) :  0.0000011
     
  
.. figure:: ex_1d_step_Test_test_rmtc.png
  :scale: 80 %
  :align: center
