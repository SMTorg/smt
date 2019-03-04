Boeing 777 engine data set
==========================

.. code-block:: python

  import numpy as np
  import os
  
  def get_b777_engine():
      this_dir = os.path.split(__file__)[0]
  
      nt = 12 * 11 * 8
      xt = np.loadtxt(os.path.join(this_dir, 'b777_engine_inputs.dat')).reshape((nt, 3))
      yt = np.loadtxt(os.path.join(this_dir, 'b777_engine_outputs.dat')).reshape((nt, 2))
      dyt_dxt = np.loadtxt(os.path.join(this_dir, 'b777_engine_derivs.dat')).reshape((nt, 2, 3))
  
      xlimits = np.array([
          [0, 0.9],
          [0, 15],
          [0, 1.],
      ])
  
      return xt, yt, dyt_dxt, xlimits
  
  
  def plot_b777_engine(xt, yt, limits, interp):
      import numpy as np
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
  
      val_M = np.array([
          0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
          0.7, 0.75, 0.8, 0.85, 0.9]) # 12
      val_h = np.array([
          0., 0.6096, 1.524, 3.048, 4.572, 6.096,
          7.62, 9.144, 10.668, 11.8872, 13.1064]) # 11
      val_t = np.array([
          0.05, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0]) # 8
  
      def get_pts(xt, yt, iy, ind_M=None, ind_h=None, ind_t=None):
          eps = 1e-5
  
          if ind_M is not None:
              M = val_M[ind_M]
              keep = abs(xt[:, 0] - M) < eps
              xt = xt[keep, :]
              yt = yt[keep, :]
          if ind_h is not None:
              h = val_h[ind_h]
              keep = abs(xt[:, 1] - h) < eps
              xt = xt[keep, :]
              yt = yt[keep, :]
          if ind_t is not None:
              t = val_t[ind_t]
              keep = abs(xt[:, 2] - t) < eps
              xt = xt[keep, :]
              yt = yt[keep, :]
  
          if ind_M is None:
              data = xt[:, 0], yt[:, iy]
          elif ind_h is None:
              data = xt[:, 1], yt[:, iy]
          elif ind_t is None:
              data = xt[:, 2], yt[:, iy]
  
          if iy == 0:
              data = data[0], data[1] / 1e6
          elif iy == 1:
              data = data[0], data[1] / 1e-4
  
          return data
  
      num = 100
      x = np.zeros((num, 3))
      lins_M = np.linspace(0., 0.9, num)
      lins_h = np.linspace(0., 13.1064, num)
      lins_t = np.linspace(0.05, 1., num)
  
      def get_x(ind_M=None, ind_h=None, ind_t=None):
          x = np.zeros((num, 3))
          x[:, 0] = lins_M
          x[:, 1] = lins_h
          x[:, 2] = lins_t
          if ind_M:
              x[:, 0] = val_M[ind_M]
          if ind_h:
              x[:, 1] = val_h[ind_h]
          if ind_t:
              x[:, 2] = val_t[ind_t]
          return x
  
      nrow = 6
      ncol = 2
  
      ind_M_1 = -2
      ind_M_2 = -5
  
      ind_t_1 = 1
      ind_t_2 = -1
  
      plt.close()
      plt.figure(figsize=(15, 25))
      plt.subplots_adjust(hspace=.5)
  
      # --------------------
  
      plt.subplot(nrow, ncol, 1)
      plt.title('M={}'.format(val_M[ind_M_1]))
      plt.xlabel('throttle')
      plt.ylabel('thrust (x 1e6 N)')
  
      plt.subplot(nrow, ncol, 2)
      plt.title('M={}'.format(val_M[ind_M_1]))
      plt.xlabel('throttle')
      plt.ylabel('SFC (x 1e-3 N/N/s)')
  
      plt.subplot(nrow, ncol, 3)
      plt.title('M={}'.format(val_M[ind_M_2]))
      plt.xlabel('throttle')
      plt.ylabel('thrust (x 1e6 N)')
  
      plt.subplot(nrow, ncol, 4)
      plt.title('M={}'.format(val_M[ind_M_2]))
      plt.xlabel('throttle')
      plt.ylabel('SFC (x 1e-3 N/N/s)')
  
      # --------------------
  
      plt.subplot(nrow, ncol, 5)
      plt.title('throttle={}'.format(val_t[ind_t_1]))
      plt.xlabel('altitude (km)')
      plt.ylabel('thrust (x 1e6 N)')
  
      plt.subplot(nrow, ncol, 6)
      plt.title('throttle={}'.format(val_t[ind_t_1]))
      plt.xlabel('altitude (km)')
      plt.ylabel('SFC (x 1e-3 N/N/s)')
  
      plt.subplot(nrow, ncol, 7)
      plt.title('throttle={}'.format(val_t[ind_t_2]))
      plt.xlabel('altitude (km)')
      plt.ylabel('thrust (x 1e6 N)')
  
      plt.subplot(nrow, ncol, 8)
      plt.title('throttle={}'.format(val_t[ind_t_2]))
      plt.xlabel('altitude (km)')
      plt.ylabel('SFC (x 1e-3 N/N/s)')
  
      # --------------------
  
      plt.subplot(nrow, ncol,  9)
      plt.title('throttle={}'.format(val_t[ind_t_1]))
      plt.xlabel('Mach number')
      plt.ylabel('thrust (x 1e6 N)')
  
      plt.subplot(nrow, ncol, 10)
      plt.title('throttle={}'.format(val_t[ind_t_1]))
      plt.xlabel('Mach number')
      plt.ylabel('SFC (x 1e-3 N/N/s)')
  
      plt.subplot(nrow, ncol, 11)
      plt.title('throttle={}'.format(val_t[ind_t_2]))
      plt.xlabel('Mach number')
      plt.ylabel('thrust (x 1e6 N)')
  
      plt.subplot(nrow, ncol, 12)
      plt.title('throttle={}'.format(val_t[ind_t_2]))
      plt.xlabel('Mach number')
      plt.ylabel('SFC (x 1e-3 N/N/s)')
  
      ind_h_list = [0, 4, 7, 10]
      ind_h_list = [4, 7, 10]
  
      ind_M_list = [0, 3, 6, 11]
      ind_M_list = [3, 6, 11]
  
      colors = ['b', 'r', 'g', 'c', 'm']
  
      # -----------------------------------------------------------------------------
  
      # Throttle slices
      for k, ind_h in enumerate(ind_h_list):
          ind_M = ind_M_1
          x = get_x(ind_M=ind_M, ind_h=ind_h)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 1)
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_h=ind_h)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_t, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 2)
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_h=ind_h)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_t, y[:, 1] / 1e-4, colors[k])
  
          ind_M = ind_M_2
          x = get_x(ind_M=ind_M, ind_h=ind_h)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 3)
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_h=ind_h)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_t, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 4)
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_h=ind_h)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_t, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      # Altitude slices
      for k, ind_M in enumerate(ind_M_list):
          ind_t = ind_t_1
          x = get_x(ind_M=ind_M, ind_t=ind_t)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 5)
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_t=ind_t)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_h, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 6)
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_t=ind_t)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_h, y[:, 1] / 1e-4, colors[k])
  
          ind_t = ind_t_2
          x = get_x(ind_M=ind_M, ind_t=ind_t)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 7)
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_t=ind_t)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_h, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 8)
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_t=ind_t)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_h, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      # Mach number slices
      for k, ind_h in enumerate(ind_h_list):
          ind_t = ind_t_1
          x = get_x(ind_t=ind_t, ind_h=ind_h)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol,  9)
          xt_, yt_ = get_pts(xt, yt, 0, ind_h=ind_h, ind_t=ind_t)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_M, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 10)
          xt_, yt_ = get_pts(xt, yt, 1, ind_h=ind_h, ind_t=ind_t)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_M, y[:, 1] / 1e-4, colors[k])
  
          ind_t = ind_t_2
          x = get_x(ind_t=ind_t, ind_h=ind_h)
          y = interp.predict_values(x)
  
          plt.subplot(nrow, ncol, 11)
          xt_, yt_ = get_pts(xt, yt, 0, ind_h=ind_h, ind_t=ind_t)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_M, y[:, 0] / 1e6, colors[k])
          plt.subplot(nrow, ncol, 12)
          xt_, yt_ = get_pts(xt, yt, 1, ind_h=ind_h, ind_t=ind_t)
          plt.plot(xt_, yt_, 'o' + colors[k])
          plt.plot(lins_M, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      for k in range(4):
          legend_entries = []
          for ind_h in ind_h_list:
              legend_entries.append('h={}'.format(val_h[ind_h]))
              legend_entries.append('')
  
          plt.subplot(nrow, ncol, k + 1)
          plt.legend(legend_entries)
  
          plt.subplot(nrow, ncol, k + 9)
          plt.legend(legend_entries)
  
          legend_entries = []
          for ind_M in ind_M_list:
              legend_entries.append('M={}'.format(val_M[ind_M]))
              legend_entries.append('')
  
          plt.subplot(nrow, ncol, k + 5)
          plt.legend(legend_entries)
  
      plt.show()
  

RMTB
----

.. code-block:: python

  from smt.surrogate_models import RMTB
  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  
  xt, yt, dyt_dxt, xlimits = get_b777_engine()
  
  interp = RMTB(num_ctrl_pts=15, xlimits=xlimits, nonlinear_maxiter=20, approx_order=2,
      energy_weight=0e-14, regularization_weight=0e-18, extrapolate=True,
  )
  interp.set_training_values(xt, yt)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
  interp.train()
  
  plot_b777_engine(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTB
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 1056
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0000029
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0004840
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.3237760
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0172501
        Pre-computing matrices - done. Time (sec):  0.3415968
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.275872076e+05 7.014980388e+09
              Solving for output 0 - done. Time (sec):  0.0974190
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.242232475e-03 3.529813372e-07
              Solving for output 1 - done. Time (sec):  0.1117461
           Solving initial startup problem (n=3375) - done. Time (sec):  0.2093480
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.275872076e+05 7.014980388e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.288085134e+04 1.953155630e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.926070874e+04 5.635278864e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.679856063e+04 3.897611903e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.383510086e+04 3.787123178e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.589886846e+04 3.284192662e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.845592605e+04 3.023670809e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.769372828e+04 2.681329751e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.530447070e+04 2.242731783e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 9.881826932e+03 2.023427168e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.588522663e+04 1.864877083e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 7.425937050e+03 1.768325991e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 6.171437551e+03 1.679045359e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.069917671e+04 1.620825061e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.653875897e+03 1.586612281e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 3.162616646e+03 1.574698211e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.260390990e+03 1.550613438e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 2.544518278e+03 1.518997468e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.438954004e+03 1.497917708e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.020600794e+03 1.492028567e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.472982211e+03 1.488930748e+08
              Solving for output 0 - done. Time (sec):  2.0936761
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.242232475e-03 3.529813372e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.291537296e-04 6.204977989e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.793575829e-04 1.823401328e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.067622624e-04 8.324728593e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.663771237e-04 7.723689681e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.161962999e-04 6.778016439e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.410753454e-04 5.166822345e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.021446522e-05 2.975869191e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.893738412e-05 2.106018741e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 3.462947412e-05 1.806092003e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 3.829940623e-05 1.704500814e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.893554551e-05 1.589471783e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.455492027e-05 1.434957749e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.181000189e-05 1.306210742e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.801702275e-05 1.256716689e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 9.540420207e-06 1.228605583e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 8.923156659e-06 1.214274817e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.032728857e-05 1.187418654e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.197810779e-05 1.156131278e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 5.640649200e-06 1.139675008e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 5.800155030e-06 1.138464239e-09
              Solving for output 1 - done. Time (sec):  2.0266397
           Solving nonlinear problem (n=3375) - done. Time (sec):  4.1204441
        Solving for degrees of freedom - done. Time (sec):  4.3298891
     Training - done. Time (sec):  4.6723299
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013630
     
     Prediction time/pt. (sec) :  0.0000136
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013299
     
     Prediction time/pt. (sec) :  0.0000133
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013170
     
     Prediction time/pt. (sec) :  0.0000132
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013120
     
     Prediction time/pt. (sec) :  0.0000131
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013528
     
     Prediction time/pt. (sec) :  0.0000135
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013430
     
     Prediction time/pt. (sec) :  0.0000134
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013108
     
     Prediction time/pt. (sec) :  0.0000131
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013711
     
     Prediction time/pt. (sec) :  0.0000137
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013142
     
     Prediction time/pt. (sec) :  0.0000131
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013907
     
     Prediction time/pt. (sec) :  0.0000139
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013778
     
     Prediction time/pt. (sec) :  0.0000138
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013249
     
     Prediction time/pt. (sec) :  0.0000132
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013211
     
     Prediction time/pt. (sec) :  0.0000132
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013158
     
     Prediction time/pt. (sec) :  0.0000132
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013251
     
     Prediction time/pt. (sec) :  0.0000133
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013161
     
     Prediction time/pt. (sec) :  0.0000132
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013239
     
     Prediction time/pt. (sec) :  0.0000132
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013199
     
     Prediction time/pt. (sec) :  0.0000132
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.surrogate_models import RMTC
  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  
  xt, yt, dyt_dxt, xlimits = get_b777_engine()
  
  interp = RMTC(num_elements=6, xlimits=xlimits, nonlinear_maxiter=20, approx_order=2,
      energy_weight=0., regularization_weight=0., extrapolate=True,
  )
  interp.set_training_values(xt, yt)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 0], 0)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 1], 1)
  interp.set_training_derivatives(xt, dyt_dxt[:, :, 2], 2)
  interp.train()
  
  plot_b777_engine(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTC
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 1056
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0447271
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0007963
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.2354128
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0929790
        Pre-computing matrices - done. Time (sec):  0.3740549
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.953712880e+05 2.069328234e+09
              Solving for output 0 - done. Time (sec):  0.2362192
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.275388330e-03 1.302956371e-07
              Solving for output 1 - done. Time (sec):  0.2478859
           Solving initial startup problem (n=2744) - done. Time (sec):  0.4842329
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.953712880e+05 2.069328234e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 3.395473455e+04 4.207215504e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 1.898057870e+04 3.530517917e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 2.333240598e+04 3.497898902e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 8.681814863e+03 3.371469368e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 4.949819209e+03 3.327449374e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 4.142525667e+03 3.320852182e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.615506471e+03 3.313017652e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.089449696e+03 3.307301940e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.989902711e+03 3.304710768e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.307749913e+03 3.303430512e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.199944141e+03 3.301967010e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.205072492e+03 3.301180433e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.101622862e+02 3.299989728e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 7.537286535e+02 3.299004383e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.663092108e+02 3.298366414e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 4.307948643e+02 3.298236430e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.592536814e+02 3.298135702e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 7.726849619e+02 3.298035108e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 3.380611827e+02 3.297995293e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.817003520e+02 3.297983120e+08
              Solving for output 0 - done. Time (sec):  4.2413142
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.275388330e-03 1.302956371e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.832007159e-04 9.426409582e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.508984452e-04 7.729954684e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.543140615e-04 5.931748882e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 8.738578779e-05 4.264799039e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 7.464142202e-05 4.050483970e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.034584861e-04 3.727135870e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.670538969e-05 3.356534089e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 5.639069218e-05 3.198217955e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 3.083720521e-05 3.117497496e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.790938957e-05 3.064016349e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.171817340e-05 3.033694346e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 3.829330773e-05 3.031427808e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.645938975e-05 3.014524373e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.123268408e-05 2.981939388e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 9.378995054e-06 2.938540797e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.213840350e-05 2.927401092e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 7.506282923e-06 2.924181088e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 8.434656892e-06 2.921961435e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 6.432195527e-06 2.919026078e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 8.888386474e-06 2.918237782e-09
              Solving for output 1 - done. Time (sec):  4.3199339
           Solving nonlinear problem (n=2744) - done. Time (sec):  8.5613511
        Solving for degrees of freedom - done. Time (sec):  9.0456772
     Training - done. Time (sec):  9.4240150
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0043721
     
     Prediction time/pt. (sec) :  0.0000437
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0039928
     
     Prediction time/pt. (sec) :  0.0000399
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0060408
     
     Prediction time/pt. (sec) :  0.0000604
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0043640
     
     Prediction time/pt. (sec) :  0.0000436
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0032110
     
     Prediction time/pt. (sec) :  0.0000321
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0044589
     
     Prediction time/pt. (sec) :  0.0000446
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027461
     
     Prediction time/pt. (sec) :  0.0000275
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0029178
     
     Prediction time/pt. (sec) :  0.0000292
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0047009
     
     Prediction time/pt. (sec) :  0.0000470
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0042703
     
     Prediction time/pt. (sec) :  0.0000427
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0035312
     
     Prediction time/pt. (sec) :  0.0000353
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0038779
     
     Prediction time/pt. (sec) :  0.0000388
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0032771
     
     Prediction time/pt. (sec) :  0.0000328
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0032148
     
     Prediction time/pt. (sec) :  0.0000321
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0032203
     
     Prediction time/pt. (sec) :  0.0000322
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0032408
     
     Prediction time/pt. (sec) :  0.0000324
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0032918
     
     Prediction time/pt. (sec) :  0.0000329
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030711
     
     Prediction time/pt. (sec) :  0.0000307
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
