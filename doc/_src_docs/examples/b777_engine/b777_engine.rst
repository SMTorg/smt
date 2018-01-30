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
           Computing dof2coeff - done. Time (sec):  0.0000000
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.3429999
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0160000
        Pre-computing matrices - done. Time (sec):  0.3590000
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.279683835e+05 7.014636029e+09
              Solving for output 0 - done. Time (sec):  0.1090000
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.561629020e-03 3.478957365e-07
              Solving for output 1 - done. Time (sec):  0.0940001
           Solving initial startup problem (n=3375) - done. Time (sec):  0.2030001
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.279683835e+05 7.014636029e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.290771875e+04 1.937717703e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.926796102e+04 5.666158017e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.503679666e+04 3.952037317e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.167512947e+04 3.855966680e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.342451353e+04 3.342799022e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.603440151e+04 3.067694601e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.997803362e+04 2.705946857e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 9.819839653e+03 2.251959770e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.003339232e+04 2.032984471e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.183291148e+04 1.879670109e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 7.808789052e+03 1.774638970e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 5.033263564e+03 1.676222940e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 5.714887011e+03 1.634427827e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.870380309e+03 1.613289645e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 7.294469961e+03 1.605967791e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 2.818235406e+03 1.568679947e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 3.503694034e+03 1.538115713e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.257902156e+03 1.505934494e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.140867345e+03 1.492991014e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.961197841e+03 1.492390080e+08
              Solving for output 0 - done. Time (sec):  2.1059999
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.561629020e-03 3.478957365e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.769745869e-04 6.184805636e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.918241145e-04 1.808721220e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.546256557e-04 8.319885662e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 2.001324861e-04 7.644575701e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.270398142e-04 6.625333832e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.120858973e-04 5.044860353e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.065457846e-05 2.923747324e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 4.770416784e-05 2.086314938e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 2.313479689e-05 1.802919981e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.249630111e-05 1.705812865e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.197919335e-05 1.590991690e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.761973483e-05 1.430487211e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.177852096e-05 1.298476033e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.535920110e-05 1.260527456e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.035998541e-05 1.241831850e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.471862001e-05 1.237146881e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 6.693770581e-06 1.206963737e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.190859614e-05 1.172063234e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 4.358113108e-06 1.143677109e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 4.731273643e-06 1.143292418e-09
              Solving for output 1 - done. Time (sec):  2.1059999
           Solving nonlinear problem (n=3375) - done. Time (sec):  4.2119999
        Solving for degrees of freedom - done. Time (sec):  4.4150000
     Training - done. Time (sec):  4.7900000
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0160000
     
     Prediction time/pt. (sec) :  0.0001600
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
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
           Computing dof2coeff - done. Time (sec):  0.0309999
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.2810001
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.1090000
        Pre-computing matrices - done. Time (sec):  0.4210000
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.974203585e+05 2.068915423e+09
              Solving for output 0 - done. Time (sec):  0.2340000
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.443820335e-03 1.318971474e-07
              Solving for output 1 - done. Time (sec):  0.2340000
           Solving initial startup problem (n=2744) - done. Time (sec):  0.4679999
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.974203585e+05 2.068915423e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.688155258e+04 4.218911755e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 2.031092063e+04 3.526075225e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 2.630440031e+04 3.498398763e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 8.925360620e+03 3.371566321e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 5.229073101e+03 3.327064514e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.200507737e+03 3.320134607e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.673119661e+03 3.312263701e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.127494399e+03 3.307018604e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.680121102e+03 3.304679152e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.942551657e+03 3.303508794e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.630281608e+02 3.302068441e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.190665125e+03 3.301279941e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 6.661237510e+02 3.300065923e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.073615692e+03 3.298964211e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 3.821971110e+02 3.298262584e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 2.888038483e+02 3.298157104e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 4.802401467e+02 3.298100255e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 4.106160799e+02 3.298069876e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 4.181679067e+02 3.297999340e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 3.669873911e+02 3.297946734e+08
              Solving for output 0 - done. Time (sec):  4.7580001
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.443820335e-03 1.318971474e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 4.038579959e-04 9.557652492e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.875650841e-04 7.875811007e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 3.102680887e-04 6.049499802e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 9.584730716e-05 4.310399863e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 8.510627077e-05 4.077413732e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 6.590238272e-05 3.756774147e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 5.306399542e-05 3.370010916e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.428443683e-05 3.206238336e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 5.807917090e-05 3.124011756e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.766247918e-05 3.067014959e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.047044207e-05 3.038847884e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 3.358512223e-05 3.035049521e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.616027333e-05 3.014745113e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.095865320e-05 2.987071743e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 8.726165254e-06 2.946599299e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 7.617103960e-06 2.931700456e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 8.510130331e-06 2.929074431e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.245956743e-05 2.927000373e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 8.113233971e-06 2.925195349e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.471545609e-05 2.921427760e-09
              Solving for output 1 - done. Time (sec):  4.7739999
           Solving nonlinear problem (n=2744) - done. Time (sec):  9.5320001
        Solving for degrees of freedom - done. Time (sec): 10.0000000
     Training - done. Time (sec): 10.4519999
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0150001
     
     Prediction time/pt. (sec) :  0.0001500
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0160000
     
     Prediction time/pt. (sec) :  0.0001600
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0159998
     
     Prediction time/pt. (sec) :  0.0001600
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
