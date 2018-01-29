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
           Computing dof2coeff - done. Time (sec):  0.0000029
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003459
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0011442
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0004232
        Pre-computing matrices - done. Time (sec):  0.0019679
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 6.450151056e-08 2.293337503e-13
              Solving for output 0 - done. Time (sec):  0.0094168
           Solving initial startup problem (n=100) - done. Time (sec):  0.0094779
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.427045627e-11 2.251640048e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.257474689e-11 2.223873525e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.527766832e-10 1.381388412e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.176067564e-10 9.187667736e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.344038473e-11 2.409299642e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.645101063e-11 1.117308377e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.711616290e-11 1.014238662e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 4.602746505e-12 8.717466081e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.070356465e-12 8.483004713e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 2.374034444e-13 8.458996377e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.228815924e-13 8.457713147e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.398932254e-13 8.455807583e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 4.076116589e-14 8.453722839e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.160000153e-14 8.453343630e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.327434935e-15 8.453284269e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.209447582e-15 8.453274332e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 2.197280242e-16 8.453270944e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 2.234235578e-16 8.453270515e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.234299803e-16 8.453270515e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.234300301e-16 8.453270515e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.234300301e-16 8.453270515e-15
              Solving for output 0 - done. Time (sec):  0.1236260
           Solving nonlinear problem (n=100) - done. Time (sec):  0.1236680
        Solving for degrees of freedom - done. Time (sec):  0.1332002
     Training - done. Time (sec):  0.1357083
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003879
     
     Prediction time/pt. (sec) :  0.0000008
     
  
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
           Computing dof2coeff - done. Time (sec):  0.0006661
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0002701
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0011249
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0004799
        Pre-computing matrices - done. Time (sec):  0.0025961
        Solving for degrees of freedom ...
           Solving initial startup problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.470849329e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.247786014e-10 2.493640129e-14
              Solving for output 0 - done. Time (sec):  0.0091050
           Solving initial startup problem (n=82) - done. Time (sec):  0.0091622
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.484040934e-12 2.493593876e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032474441e-12 2.483229615e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.524071904e-11 2.371029299e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 4.548381717e-11 1.697968148e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 4.289197076e-11 1.655363158e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 1.484067476e-11 1.210235795e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 4.214372513e-12 1.120623342e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 6.777355478e-13 1.109822728e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.555802039e-13 1.109144182e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 8.196688748e-14 1.109022069e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.228229921e-14 1.108951755e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 5.460275386e-15 1.108941513e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 9.421205444e-16 1.108940402e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 7.950416640e-17 1.108940342e-14
              Solving for output 0 - done. Time (sec):  0.0884631
           Solving nonlinear problem (n=82) - done. Time (sec):  0.0885131
        Solving for degrees of freedom - done. Time (sec):  0.0977290
     Training - done. Time (sec):  0.1006501
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004971
     
     Prediction time/pt. (sec) :  0.0000010
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center
