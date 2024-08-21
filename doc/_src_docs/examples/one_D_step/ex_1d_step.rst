1-D step-like data set
======================

.. code-block:: python

  import numpy as np
  
  
  def get_one_d_step():
      xt = np.array(
          [
              0.0000,
              0.4000,
              0.6000,
              0.7000,
              0.7500,
              0.7750,
              0.8000,
              0.8500,
              0.8750,
              0.9000,
              0.9250,
              0.9500,
              0.9750,
              1.0000,
              1.0250,
              1.0500,
              1.1000,
              1.2000,
              1.3000,
              1.4000,
              1.6000,
              1.8000,
              2.0000,
          ],
          dtype=np.float64,
      )
      yt = np.array(
          [
              0.0130,
              0.0130,
              0.0130,
              0.0130,
              0.0130,
              0.0130,
              0.0130,
              0.0132,
              0.0135,
              0.0140,
              0.0162,
              0.0230,
              0.0275,
              0.0310,
              0.0344,
              0.0366,
              0.0396,
              0.0410,
              0.0403,
              0.0390,
              0.0360,
              0.0350,
              0.0345,
          ],
          dtype=np.float64,
      )
  
      xlimits = np.array([[0.0, 2.0]])
  
      return xt, yt, xlimits
  
  
  def plot_one_d_step(xt, yt, limits, interp):
      import matplotlib
      import numpy as np
  
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
  
      num = 500
      x = np.linspace(0.0, 2.0, num)
      y = interp.predict_values(x)[:, 0]
  
      plt.plot(x, y)
      plt.plot(xt, yt, "o")
      plt.xlabel("x")
      plt.ylabel("y")
      plt.show()
  

RMTB
----

.. code-block:: python

  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  from smt.surrogate_models import RMTB
  
  xt, yt, xlimits = get_one_d_step()
  
  interp = RMTB(
      num_ctrl_pts=100,
      xlimits=xlimits,
      nonlinear_maxiter=20,
      solver_tolerance=1e-16,
      energy_weight=1e-14,
      regularization_weight=0.0,
  )
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
           Computing energy terms - done. Time (sec):  0.0000000
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.0000000
        Solving for degrees of freedom ...
           Solving initial startup problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.032652876e-01 8.436300000e-03
                 Iteration (num., iy, grad. norm, func.) :   0   0 5.558767804e-08 2.270128086e-13
              Solving for output 0 - done. Time (sec):  0.0159578
           Solving initial startup problem (n=100) - done. Time (sec):  0.0159578
           Solving nonlinear problem (n=100) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.442278272e-11 2.234025428e-13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.276932427e-11 2.206468481e-13
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.604333797e-10 1.412945915e-13
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.274195007e-10 9.555539967e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.638115965e-11 2.479560025e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.789323778e-11 1.138389897e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 2.205371728e-11 1.075389070e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 6.005987821e-12 8.818284843e-15
                 Iteration (num., iy, grad. norm, func.) :   7   0 4.105104107e-12 8.682287242e-15
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.166159130e-12 8.497521242e-15
                 Iteration (num., iy, grad. norm, func.) :   9   0 3.907475214e-13 8.466958190e-15
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.123639852e-13 8.455386478e-15
                 Iteration (num., iy, grad. norm, func.) :  11   0 6.687210306e-14 8.454404117e-15
                 Iteration (num., iy, grad. norm, func.) :  12   0 3.060099646e-14 8.453844993e-15
                 Iteration (num., iy, grad. norm, func.) :  13   0 9.013685289e-15 8.453413582e-15
                 Iteration (num., iy, grad. norm, func.) :  14   0 3.568462456e-15 8.453316548e-15
                 Iteration (num., iy, grad. norm, func.) :  15   0 7.068428651e-16 8.453277203e-15
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.357685654e-16 8.453272016e-15
                 Iteration (num., iy, grad. norm, func.) :  17   0 4.395170225e-16 8.453271767e-15
                 Iteration (num., iy, grad. norm, func.) :  18   0 3.694965901e-16 8.453271676e-15
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.772757574e-16 8.453270829e-15
              Solving for output 0 - done. Time (sec):  0.0848029
           Solving nonlinear problem (n=100) - done. Time (sec):  0.0848029
        Solving for degrees of freedom - done. Time (sec):  0.1007607
     Training - done. Time (sec):  0.1007607
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0156202
     
     Prediction time/pt. (sec) :  0.0000312
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.examples.one_D_step.one_D_step import get_one_d_step, plot_one_d_step
  from smt.surrogate_models import RMTC
  
  xt, yt, xlimits = get_one_d_step()
  
  interp = RMTC(
      num_elements=40,
      xlimits=xlimits,
      nonlinear_maxiter=20,
      solver_tolerance=1e-16,
      energy_weight=1e-14,
      regularization_weight=0.0,
  )
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
                 Iteration (num., iy, grad. norm, func.) :   0   0 5.814481774e-10 2.493602350e-14
              Solving for output 0 - done. Time (sec):  0.0159605
           Solving initial startup problem (n=82) - done. Time (sec):  0.0159605
           Solving nonlinear problem (n=82) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.483954478e-12 2.493518683e-14
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.032369145e-12 2.483155071e-14
                 Iteration (num., iy, grad. norm, func.) :   1   0 8.718452192e-11 2.392752964e-14
                 Iteration (num., iy, grad. norm, func.) :   2   0 4.131175954e-11 1.658086360e-14
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.955373996e-11 1.636785165e-14
                 Iteration (num., iy, grad. norm, func.) :   4   0 1.291670646e-11 1.192880996e-14
                 Iteration (num., iy, grad. norm, func.) :   5   0 3.378763717e-12 1.116775902e-14
                 Iteration (num., iy, grad. norm, func.) :   6   0 4.769319870e-13 1.109480428e-14
                 Iteration (num., iy, grad. norm, func.) :   7   0 9.630933570e-14 1.109039946e-14
                 Iteration (num., iy, grad. norm, func.) :   8   0 5.380254088e-14 1.108982132e-14
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.228166919e-14 1.108945934e-14
                 Iteration (num., iy, grad. norm, func.) :  10   0 3.096828107e-15 1.108941229e-14
                 Iteration (num., iy, grad. norm, func.) :  11   0 8.923711312e-16 1.108940503e-14
                 Iteration (num., iy, grad. norm, func.) :  12   0 2.580481756e-16 1.108940368e-14
                 Iteration (num., iy, grad. norm, func.) :  13   0 7.019512501e-17 1.108940343e-14
              Solving for output 0 - done. Time (sec):  0.0844636
           Solving nonlinear problem (n=82) - done. Time (sec):  0.0844636
        Solving for degrees of freedom - done. Time (sec):  0.1004241
     Training - done. Time (sec):  0.1004241
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: ex_1d_step.png
  :scale: 80 %
  :align: center
