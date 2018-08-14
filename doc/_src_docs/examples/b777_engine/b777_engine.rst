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
           Computing dof2coeff - done. Time (sec):  0.0000019
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0005519
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.2675509
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0107069
        Pre-computing matrices - done. Time (sec):  0.2789130
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.311813731e+05 7.011307738e+09
              Solving for output 0 - done. Time (sec):  0.1019819
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.257433972e-03 3.486517486e-07
              Solving for output 1 - done. Time (sec):  0.1028950
           Solving initial startup problem (n=3375) - done. Time (sec):  0.2049861
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.311813731e+05 7.011307738e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.965120516e+04 1.951366686e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 5.396838148e+04 5.605012425e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.714092287e+04 3.879531692e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.280718047e+04 3.774764849e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.503356073e+04 3.277714567e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.784173795e+04 3.019176057e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.546319674e+04 2.678745024e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.506278392e+04 2.233839224e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.016190969e+04 2.016216461e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.324881561e+04 1.855062300e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 6.593214152e+03 1.767151650e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 5.916305145e+03 1.691645006e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 4.898814761e+03 1.615949441e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.370184955e+03 1.573748134e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 2.865402722e+03 1.567607869e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 4.600477945e+03 1.553048647e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 3.229406560e+03 1.522107667e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.733517617e+03 1.494924546e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 9.327453160e+02 1.482813596e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 9.296330730e+02 1.482257906e+08
              Solving for output 0 - done. Time (sec):  2.0591099
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.257433972e-03 3.486517486e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.856865310e-04 6.188762158e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.090987605e-04 1.805393267e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.292527953e-04 8.395367884e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.867967907e-04 7.695392525e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.194913355e-04 6.597867459e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.257729305e-04 4.977259267e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 3.730861320e-05 2.892980134e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.777492631e-05 2.075282056e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 3.563960932e-05 1.801579626e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.522035695e-05 1.711669638e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.950133456e-05 1.609480271e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.345453568e-05 1.437894236e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.318135967e-05 1.299443888e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.049897484e-05 1.260520813e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.109435302e-05 1.244196587e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.377604133e-05 1.236753171e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 8.669143722e-06 1.202473958e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 9.298067203e-06 1.171125459e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 4.215327340e-06 1.146620553e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 4.857774033e-06 1.144494235e-09
              Solving for output 1 - done. Time (sec):  2.0684009
           Solving nonlinear problem (n=3375) - done. Time (sec):  4.1276140
        Solving for degrees of freedom - done. Time (sec):  4.3327200
     Training - done. Time (sec):  4.6231680
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0015130
     
     Prediction time/pt. (sec) :  0.0000151
     
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
     Predicting - done. Time (sec):  0.0013559
     
     Prediction time/pt. (sec) :  0.0000136
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013280
     
     Prediction time/pt. (sec) :  0.0000133
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013168
     
     Prediction time/pt. (sec) :  0.0000132
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012989
     
     Prediction time/pt. (sec) :  0.0000130
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013092
     
     Prediction time/pt. (sec) :  0.0000131
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012960
     
     Prediction time/pt. (sec) :  0.0000130
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012970
     
     Prediction time/pt. (sec) :  0.0000130
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013011
     
     Prediction time/pt. (sec) :  0.0000130
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012932
     
     Prediction time/pt. (sec) :  0.0000129
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013011
     
     Prediction time/pt. (sec) :  0.0000130
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012910
     
     Prediction time/pt. (sec) :  0.0000129
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012939
     
     Prediction time/pt. (sec) :  0.0000129
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013139
     
     Prediction time/pt. (sec) :  0.0000131
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013092
     
     Prediction time/pt. (sec) :  0.0000131
     
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
     Predicting - done. Time (sec):  0.0013092
     
     Prediction time/pt. (sec) :  0.0000131
     
  
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
           Computing dof2coeff - done. Time (sec):  0.0326629
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0007999
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.2017739
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0771909
        Pre-computing matrices - done. Time (sec):  0.3125899
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.091165474e+05 2.064092003e+09
              Solving for output 0 - done. Time (sec):  0.2094460
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.273593250e-03 1.302915302e-07
              Solving for output 1 - done. Time (sec):  0.2099199
           Solving initial startup problem (n=2744) - done. Time (sec):  0.4195151
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.091165474e+05 2.064092003e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.872734602e+04 4.215431454e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 1.620453365e+04 3.528011012e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 2.430657834e+04 3.499429321e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.154308832e+04 3.371347377e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 4.740115637e+03 3.327128462e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 4.597555421e+03 3.320556229e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.811164331e+03 3.312669549e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.025828684e+03 3.307184840e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.632956618e+03 3.304715340e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.049605605e+03 3.303574204e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.977705286e+02 3.302163395e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.295393001e+03 3.301320075e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.025003589e+03 3.299979263e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 8.553274018e+02 3.299032885e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 5.698722572e+02 3.298450940e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 6.244963527e+02 3.298398752e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.878335459e+02 3.298394675e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 9.066021711e+02 3.298342491e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 5.046763511e+02 3.298197510e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 3.720231239e+02 3.298157221e+08
              Solving for output 0 - done. Time (sec):  4.2596631
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.273593250e-03 1.302915302e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.798704756e-04 9.432578883e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.137192458e-04 7.782210088e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.471648601e-04 5.960573013e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 8.415077368e-05 4.271145427e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 9.216505380e-05 4.049548965e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 5.770085665e-05 3.728981792e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.472873377e-05 3.361206907e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.502211935e-05 3.204253937e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 3.509945228e-05 3.123182466e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 3.724749689e-05 3.064712589e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.411417545e-05 3.035324139e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 1.790127573e-05 3.019959957e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.969293720e-05 3.000300631e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.849368897e-05 2.974685229e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.343940085e-05 2.953147620e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.118951597e-05 2.936924251e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 8.458545619e-06 2.928980247e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.469855830e-05 2.926635046e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 9.899723266e-06 2.924276223e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.046040073e-05 2.923617330e-09
              Solving for output 1 - done. Time (sec):  4.2580571
           Solving nonlinear problem (n=2744) - done. Time (sec):  8.5178111
        Solving for degrees of freedom - done. Time (sec):  8.9374180
     Training - done. Time (sec):  9.2640190
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028751
     
     Prediction time/pt. (sec) :  0.0000288
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027599
     
     Prediction time/pt. (sec) :  0.0000276
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027828
     
     Prediction time/pt. (sec) :  0.0000278
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027740
     
     Prediction time/pt. (sec) :  0.0000277
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028071
     
     Prediction time/pt. (sec) :  0.0000281
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028100
     
     Prediction time/pt. (sec) :  0.0000281
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0026181
     
     Prediction time/pt. (sec) :  0.0000262
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0025420
     
     Prediction time/pt. (sec) :  0.0000254
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027599
     
     Prediction time/pt. (sec) :  0.0000276
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0026600
     
     Prediction time/pt. (sec) :  0.0000266
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0026901
     
     Prediction time/pt. (sec) :  0.0000269
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0026619
     
     Prediction time/pt. (sec) :  0.0000266
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027571
     
     Prediction time/pt. (sec) :  0.0000276
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0026581
     
     Prediction time/pt. (sec) :  0.0000266
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027869
     
     Prediction time/pt. (sec) :  0.0000279
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027111
     
     Prediction time/pt. (sec) :  0.0000271
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0028200
     
     Prediction time/pt. (sec) :  0.0000282
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0027809
     
     Prediction time/pt. (sec) :  0.0000278
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
