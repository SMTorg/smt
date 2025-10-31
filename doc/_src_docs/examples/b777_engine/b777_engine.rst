Boeing 777 engine data set
==========================

.. code-block:: python

  import os
  
  import numpy as np
  
  
  def get_b777_engine():
      this_dir = os.path.split(__file__)[0]
  
      nt = 12 * 11 * 8
      xt = np.loadtxt(os.path.join(this_dir, "b777_engine_inputs.dat")).reshape((nt, 3))
      yt = np.loadtxt(os.path.join(this_dir, "b777_engine_outputs.dat")).reshape((nt, 2))
      dyt_dxt = np.loadtxt(os.path.join(this_dir, "b777_engine_derivs.dat")).reshape(
          (nt, 2, 3)
      )
  
      xlimits = np.array([[0, 0.9], [0, 15], [0, 1.0]])
  
      return xt, yt, dyt_dxt, xlimits
  
  
  def plot_b777_engine(xt, yt, limits, interp):
      import matplotlib
      import numpy as np
  
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
  
      val_M = np.array(
          [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
      )  # 12
      val_h = np.array(
          [0.0, 0.6096, 1.524, 3.048, 4.572, 6.096, 7.62, 9.144, 10.668, 11.8872, 13.1064]
      )  # 11
      val_t = np.array([0.05, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0])  # 8
  
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
      lins_M = np.linspace(0.0, 0.9, num)
      lins_h = np.linspace(0.0, 13.1064, num)
      lins_t = np.linspace(0.05, 1.0, num)
  
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
  
      # --------------------
  
      fig, axs = plt.subplots(nrow, ncol, gridspec_kw={"hspace": 0.5}, figsize=(15, 25))
  
      axs[0, 0].set_title("M={}".format(val_M[ind_M_1]))
      axs[0, 0].set(xlabel="throttle", ylabel="thrust (x 1e6 N)")
  
      axs[0, 1].set_title("M={}".format(val_M[ind_M_1]))
      axs[0, 1].set(xlabel="throttle", ylabel="SFC (x 1e-3 N/N/s)")
  
      axs[1, 0].set_title("M={}".format(val_M[ind_M_2]))
      axs[1, 0].set(xlabel="throttle", ylabel="thrust (x 1e6 N)")
  
      axs[1, 1].set_title("M={}".format(val_M[ind_M_2]))
      axs[1, 1].set(xlabel="throttle", ylabel="SFC (x 1e-3 N/N/s)")
  
      # --------------------
  
      axs[2, 0].set_title("throttle={}".format(val_t[ind_t_1]))
      axs[2, 0].set(xlabel="altitude (km)", ylabel="thrust (x 1e6 N)")
  
      axs[2, 1].set_title("throttle={}".format(val_t[ind_t_1]))
      axs[2, 1].set(xlabel="altitude (km)", ylabel="SFC (x 1e-3 N/N/s)")
  
      axs[3, 0].set_title("throttle={}".format(val_t[ind_t_2]))
      axs[3, 0].set(xlabel="altitude (km)", ylabel="thrust (x 1e6 N)")
  
      axs[3, 1].set_title("throttle={}".format(val_t[ind_t_2]))
      axs[3, 1].set(xlabel="altitude (km)", ylabel="SFC (x 1e-3 N/N/s)")
  
      # --------------------
  
      axs[4, 0].set_title("throttle={}".format(val_t[ind_t_1]))
      axs[4, 0].set(xlabel="Mach number", ylabel="thrust (x 1e6 N)")
  
      axs[4, 1].set_title("throttle={}".format(val_t[ind_t_1]))
      axs[4, 1].set(xlabel="Mach number", ylabel="SFC (x 1e-3 N/N/s)")
  
      axs[5, 0].set_title("throttle={}".format(val_t[ind_t_2]))
      axs[5, 0].set(xlabel="Mach number", ylabel="thrust (x 1e6 N)")
  
      axs[5, 1].set_title("throttle={}".format(val_t[ind_t_2]))
      axs[5, 1].set(xlabel="Mach number", ylabel="SFC (x 1e-3 N/N/s)")
  
      ind_h_list = [0, 4, 7, 10]
      ind_h_list = [4, 7, 10]
  
      ind_M_list = [0, 3, 6, 11]
      ind_M_list = [3, 6, 11]
  
      colors = ["b", "r", "g", "c", "m"]
  
      # -----------------------------------------------------------------------------
  
      # Throttle slices
      for k, ind_h in enumerate(ind_h_list):
          ind_M = ind_M_1
          x = get_x(ind_M=ind_M, ind_h=ind_h)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_h=ind_h)
          axs[0, 0].plot(xt_, yt_, "o" + colors[k])
          axs[0, 0].plot(lins_t, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_h=ind_h)
          axs[0, 1].plot(xt_, yt_, "o" + colors[k])
          axs[0, 1].plot(lins_t, y[:, 1] / 1e-4, colors[k])
  
          ind_M = ind_M_2
          x = get_x(ind_M=ind_M, ind_h=ind_h)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_h=ind_h)
          axs[1, 0].plot(xt_, yt_, "o" + colors[k])
          axs[1, 0].plot(lins_t, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_h=ind_h)
          axs[1, 1].plot(xt_, yt_, "o" + colors[k])
          axs[1, 1].plot(lins_t, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      # Altitude slices
      for k, ind_M in enumerate(ind_M_list):
          ind_t = ind_t_1
          x = get_x(ind_M=ind_M, ind_t=ind_t)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_t=ind_t)
          axs[2, 0].plot(xt_, yt_, "o" + colors[k])
          axs[2, 0].plot(lins_h, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_t=ind_t)
          axs[2, 1].plot(xt_, yt_, "o" + colors[k])
          axs[2, 1].plot(lins_h, y[:, 1] / 1e-4, colors[k])
  
          ind_t = ind_t_2
          x = get_x(ind_M=ind_M, ind_t=ind_t)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_M=ind_M, ind_t=ind_t)
          axs[3, 0].plot(xt_, yt_, "o" + colors[k])
          axs[3, 0].plot(lins_h, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_M=ind_M, ind_t=ind_t)
          axs[3, 1].plot(xt_, yt_, "o" + colors[k])
          axs[3, 1].plot(lins_h, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      # Mach number slices
      for k, ind_h in enumerate(ind_h_list):
          ind_t = ind_t_1
          x = get_x(ind_t=ind_t, ind_h=ind_h)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_h=ind_h, ind_t=ind_t)
          axs[4, 0].plot(xt_, yt_, "o" + colors[k])
          axs[4, 0].plot(lins_M, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_h=ind_h, ind_t=ind_t)
          axs[4, 1].plot(xt_, yt_, "o" + colors[k])
          axs[4, 1].plot(lins_M, y[:, 1] / 1e-4, colors[k])
  
          ind_t = ind_t_2
          x = get_x(ind_t=ind_t, ind_h=ind_h)
          y = interp.predict_values(x)
  
          xt_, yt_ = get_pts(xt, yt, 0, ind_h=ind_h, ind_t=ind_t)
          axs[5, 0].plot(xt_, yt_, "o" + colors[k])
          axs[5, 0].plot(lins_M, y[:, 0] / 1e6, colors[k])
  
          xt_, yt_ = get_pts(xt, yt, 1, ind_h=ind_h, ind_t=ind_t)
          axs[5, 1].plot(xt_, yt_, "o" + colors[k])
          axs[5, 1].plot(lins_M, y[:, 1] / 1e-4, colors[k])
  
      # -----------------------------------------------------------------------------
  
      for k in range(2):
          legend_entries = []
          for ind_h in ind_h_list:
              legend_entries.append("h={}".format(val_h[ind_h]))
              legend_entries.append("")
  
          axs[k, 0].legend(legend_entries)
          axs[k, 1].legend(legend_entries)
  
          axs[k + 4, 0].legend(legend_entries)
          axs[k + 4, 1].legend(legend_entries)
  
          legend_entries = []
          for ind_M in ind_M_list:
              legend_entries.append("M={}".format(val_M[ind_M]))
              legend_entries.append("")
  
          axs[k + 2, 0].legend(legend_entries)
          axs[k + 2, 1].legend(legend_entries)
  
      plt.show()
  

RMTB
----

.. code-block:: python

  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  from smt.surrogate_models import RMTB
  
  xt, yt, dyt_dxt, xlimits = get_b777_engine()
  
  interp = RMTB(
      num_ctrl_pts=15,
      xlimits=xlimits,
      nonlinear_maxiter=20,
      approx_order=2,
      energy_weight=0e-14,
      regularization_weight=0e-18,
      extrapolate=True,
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
           Computing energy terms - done. Time (sec):  0.2252231
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0072894
        Pre-computing matrices - done. Time (sec):  0.2325125
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.373826973e+05 6.997915387e+09
              Solving for output 0 - done. Time (sec):  0.0580795
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.395061018e-03 3.468699832e-07
              Solving for output 1 - done. Time (sec):  0.0635629
           Solving initial startup problem (n=3375) - done. Time (sec):  0.1216424
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.373826973e+05 6.997915387e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.235665692e+04 1.954806038e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.654098320e+04 5.658756761e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.672346133e+04 3.885491673e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.261616842e+04 3.768480084e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.686026706e+04 3.249773050e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.626148419e+04 2.983960747e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.524365745e+04 2.654419506e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 8.490561347e+03 2.216463966e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 9.545883104e+03 2.016764770e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 5.720345076e+03 1.864751429e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 8.662166329e+03 1.767845928e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 6.197316018e+03 1.659797529e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 4.224243819e+03 1.618341373e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 3.112670522e+03 1.600496853e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.370148466e+03 1.600262496e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 2.859520501e+03 1.569733173e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 2.782479646e+03 1.533014054e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.299670974e+03 1.496565883e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 1.610566561e+03 1.487769054e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.447300133e+03 1.485878967e+08
              Solving for output 0 - done. Time (sec):  1.1253736
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.395061018e-03 3.468699832e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.914872455e-04 6.182312112e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.865874329e-04 1.805178232e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.247617079e-04 8.253323304e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.768547031e-04 7.596092771e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.254616429e-04 6.616975834e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.014353342e-04 5.078978077e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.561000928e-05 2.963531897e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 6.361066346e-05 2.080802088e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 2.006390508e-05 1.779385831e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 1.934234213e-05 1.701506695e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.059391901e-05 1.612436453e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.588418235e-05 1.449071076e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.072301170e-05 1.307094325e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.014181444e-05 1.265540786e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.119759769e-05 1.250124472e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.353578802e-05 1.231299250e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.274579638e-05 1.185742022e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 9.960075664e-06 1.162834806e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 5.406897757e-06 1.149088640e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 8.596626113e-06 1.145548481e-09
              Solving for output 1 - done. Time (sec):  1.1621206
           Solving nonlinear problem (n=3375) - done. Time (sec):  2.2874942
        Solving for degrees of freedom - done. Time (sec):  2.4096520
     Training - done. Time (sec):  2.6421645
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
     Predicting - done. Time (sec):  0.0060663
     
     Prediction time/pt. (sec) :  0.0000607
     
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

  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  from smt.surrogate_models import RMTC
  
  xt, yt, dyt_dxt, xlimits = get_b777_engine()
  
  interp = RMTC(
      num_elements=6,
      xlimits=xlimits,
      nonlinear_maxiter=20,
      approx_order=2,
      energy_weight=0.0,
      regularization_weight=0.0,
      extrapolate=True,
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
           Computing dof2coeff - done. Time (sec):  0.0195930
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.1204863
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0347147
        Pre-computing matrices - done. Time (sec):  0.1747940
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.013823241e+05 2.067229908e+09
              Solving for output 0 - done. Time (sec):  0.1299539
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.423039559e-03 1.317122164e-07
              Solving for output 1 - done. Time (sec):  0.1213923
           Solving initial startup problem (n=2744) - done. Time (sec):  0.2513461
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.013823241e+05 2.067229908e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 3.142149731e+04 4.216257744e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 1.628819259e+04 3.528261743e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 2.431474045e+04 3.502779731e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.979393989e+03 3.372917623e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 4.504412583e+03 3.327215706e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.742398374e+03 3.320226809e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.634482225e+03 3.312305682e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.027481346e+03 3.307021257e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.164812990e+03 3.304671562e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.056698510e+03 3.303603150e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.588700733e+03 3.302176992e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.417683539e+03 3.301320906e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.464194882e+02 3.300058274e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 1.469318435e+03 3.299022133e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 3.776511399e+02 3.298368605e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 5.774489399e+02 3.298363026e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 4.650683101e+02 3.298262578e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 6.478920533e+02 3.298143199e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 3.655761629e+02 3.297990853e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 2.635230494e+02 3.297943728e+08
              Solving for output 0 - done. Time (sec):  2.5887163
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.423039559e-03 1.317122164e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 4.801149326e-04 9.499902244e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.806984163e-04 7.828673651e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.607850877e-04 6.049600013e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 9.397660769e-05 4.307701931e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 8.401306727e-05 4.061384351e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 8.858356348e-05 3.737740437e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.418680845e-05 3.360984713e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 4.194077635e-05 3.204942422e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 5.270100036e-05 3.122969531e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.806077101e-05 3.065922721e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.302063128e-05 3.043676807e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 3.443713096e-05 3.035276555e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 2.187733093e-05 3.018521234e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.908237935e-05 2.990285088e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.556075444e-05 2.957239892e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.298701588e-05 2.936695843e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 8.194155021e-06 2.928570971e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 9.208864453e-06 2.925633327e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 7.771173950e-06 2.923701376e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.062166843e-05 2.920674153e-09
              Solving for output 1 - done. Time (sec):  2.6449261
           Solving nonlinear problem (n=2744) - done. Time (sec):  5.2336423
        Solving for degrees of freedom - done. Time (sec):  5.4849885
     Training - done. Time (sec):  5.6648791
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
     Predicting - done. Time (sec):  0.0050323
     
     Prediction time/pt. (sec) :  0.0000503
     
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
     Predicting - done. Time (sec):  0.0050933
     
     Prediction time/pt. (sec) :  0.0000509
     
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
     Predicting - done. Time (sec):  0.0050542
     
     Prediction time/pt. (sec) :  0.0000505
     
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
