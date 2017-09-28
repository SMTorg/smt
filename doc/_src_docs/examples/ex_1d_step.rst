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
           Computing dof2coeff - done. Time (sec):  0.0000029
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0004122
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0011921
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0004041
        Pre-computing matrices - done. Time (sec):  0.0020611
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 6.450151056e-08 2.293337503e-13
              Solving for output 0 - done. Time (sec):  0.0061829
           Solving initial startup problem (n=100) - done. Time (sec):  0.0062289
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.427045627e-11 2.251640048e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.256597888e-11 2.223872768e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.504083181e-10 1.376597861e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.317410246e-10 9.678675309e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.757488963e-11 2.508025749e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.825498792e-11 1.144502459e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 7.853276069e-12 8.961817123e-15
                 Iteration (num., iy, grad. norm, func.) :   6   0 6.924286429e-12 8.883085681e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 5.856552607e-12 8.827022638e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.384168474e-12 8.492836075e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.072792007e-12 8.483274174e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 2.589190336e-13 8.461717840e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.553223934e-13 8.457994601e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 5.372053993e-14 8.454404735e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.394256249e-14 8.453388274e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.311454658e-14 8.453326122e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 2.290366695e-15 8.453292496e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 1.374456134e-15 8.453281451e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 1.587304137e-15 8.453273305e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 1.341225445e-16 8.453270785e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.268970282e-16 8.453270706e-15
              Solving for output 0 - done. Time (sec):  0.1904519
           Solving nonlinear problem (n=100) - done. Time (sec):  0.1904991
        Solving for degrees of freedom - done. Time (sec):  0.1967850
     Training - done. Time (sec):  0.1991789
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006199
     
     Prediction time/pt. (sec) :  0.0000012
     
  
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
           Computing dof2coeff - done. Time (sec):  0.0008740
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003021
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0011702
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0005081
        Pre-computing matrices - done. Time (sec):  0.0029120
        Solving for degrees of freedom ...
           Solving initial startup problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.470849329e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.247786014e-10 2.493640129e-14
              Solving for output 0 - done. Time (sec):  0.0063350
           Solving initial startup problem (n=82) - done. Time (sec):  0.0063941
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.484040934e-12 2.493593876e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032474441e-12 2.483229615e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.524071904e-11 2.371029299e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 4.548381717e-11 1.697968148e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 4.289197076e-11 1.655363158e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 1.484067476e-11 1.210235795e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 4.214372513e-12 1.120623342e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 6.771498846e-13 1.109821368e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.551570970e-13 1.109142780e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 8.532217095e-14 1.109025897e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.331700417e-14 1.108952450e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 5.770675015e-15 1.108941613e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.019560468e-15 1.108940408e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.513554605e-17 1.108940343e-14
              Solving for output 0 - done. Time (sec):  0.0884550
           Solving nonlinear problem (n=82) - done. Time (sec):  0.0885000
        Solving for degrees of freedom - done. Time (sec):  0.0949481
     Training - done. Time (sec):  0.0981979
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0004640
     
     Prediction time/pt. (sec) :  0.0000009
     
  
.. figure:: ex_1d_step_Test_test_rmtc.png
  :scale: 80 %
  :align: center
