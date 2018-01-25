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
           Computing dof2coeff - done. Time (sec):  0.0000031
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0003991
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0012789
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0004048
        Pre-computing matrices - done. Time (sec):  0.0021441
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 6.450151056e-08 2.293337503e-13
              Solving for output 0 - done. Time (sec):  0.0102262
           Solving initial startup problem (n=100) - done. Time (sec):  0.0102882
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.427045627e-11 2.251640048e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.257474689e-11 2.223873525e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.527766832e-10 1.381388412e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.102435443e-10 8.971318961e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.122119284e-11 2.367543049e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.636116955e-11 1.117160018e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 7.492735193e-12 8.988800842e-15
                 Iteration (num., iy, grad. norm, func.) :   6   0 6.556188649e-12 8.924265035e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.734665560e-12 8.524233650e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 4.621905004e-13 8.465121815e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 4.350788821e-13 8.464406812e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.290397961e-13 8.455715825e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 9.967641710e-14 8.455027806e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 2.885253753e-14 8.453613511e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 8.414563566e-15 8.453327728e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.536554094e-15 8.453299995e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 1.003006570e-15 8.453278569e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.385955308e-16 8.453273937e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 4.067958553e-16 8.453271301e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 4.626357432e-16 8.453271275e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.546154465e-16 8.453270676e-15
              Solving for output 0 - done. Time (sec):  0.1464331
           Solving nonlinear problem (n=100) - done. Time (sec):  0.1464798
        Solving for degrees of freedom - done. Time (sec):  0.1568251
     Training - done. Time (sec):  0.1597269
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003750
     
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
           Computing dof2coeff - done. Time (sec):  0.0019791
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0004940
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0018539
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0006952
        Pre-computing matrices - done. Time (sec):  0.0051181
        Solving for degrees of freedom ...
           Solving initial startup problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.470849329e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.247786014e-10 2.493640129e-14
              Solving for output 0 - done. Time (sec):  0.0068610
           Solving initial startup problem (n=82) - done. Time (sec):  0.0069528
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.484040934e-12 2.493593876e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032474441e-12 2.483229615e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.511790759e-11 2.369597121e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 4.554697613e-11 1.672724049e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.241231806e-11 1.168626371e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 5.288171535e-12 1.124614013e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.185179007e-12 1.110699560e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 3.547743513e-13 1.109436572e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 8.619136020e-14 1.109028035e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 2.248050262e-14 1.108952346e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 4.751251485e-15 1.108941426e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 6.956642128e-16 1.108940422e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.318707637e-16 1.108940350e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 2.218143646e-16 1.108940344e-14
                 Iteration (num., iy, grad. norm, func.) :  13   0 2.218147727e-16 1.108940344e-14
                 Iteration (num., iy, grad. norm, func.) :  14   0 2.218164055e-16 1.108940344e-14
                 Iteration (num., iy, grad. norm, func.) :  15   0 2.218164059e-16 1.108940344e-14
                 Iteration (num., iy, grad. norm, func.) :  16   0 2.218164059e-16 1.108940344e-14
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.218164059e-16 1.108940344e-14
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.218164059e-16 1.108940344e-14
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.218164059e-16 1.108940344e-14
              Solving for output 0 - done. Time (sec):  0.1071498
           Solving nonlinear problem (n=82) - done. Time (sec):  0.1072049
        Solving for degrees of freedom - done. Time (sec):  0.1142173
     Training - done. Time (sec):  0.1199369
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0003519
     
     Prediction time/pt. (sec) :  0.0000007
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center
