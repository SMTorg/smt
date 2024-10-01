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
           Computing energy terms - done. Time (sec):  0.1836782
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0108161
        Pre-computing matrices - done. Time (sec):  0.1944942
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.286579719e+05 7.013975979e+09
              Solving for output 0 - done. Time (sec):  0.0639007
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.604497664e-03 3.484939516e-07
              Solving for output 1 - done. Time (sec):  0.0644658
           Solving initial startup problem (n=3375) - done. Time (sec):  0.1283665
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.286579719e+05 7.013975979e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.469165704e+04 1.937634358e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.879509199e+04 5.667652516e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.396400747e+04 3.943728006e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.052422987e+04 3.847124210e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.707363528e+04 3.317086596e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.645948640e+04 3.031832603e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.639951693e+04 2.681480939e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 8.776758788e+03 2.237172457e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.634865326e+04 2.024493123e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 8.646698096e+03 1.871615537e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 8.194284160e+03 1.767782126e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 5.868982446e+03 1.653290075e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 4.588219094e+03 1.621900314e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 4.092068552e+03 1.612355794e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.471481177e+03 1.603871781e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.340521889e+03 1.563386070e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 2.934718911e+03 1.532056381e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.805515205e+03 1.503515957e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 3.652007244e+03 1.489115482e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.250904940e+03 1.487100656e+08
              Solving for output 0 - done. Time (sec):  1.3380342
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.604497664e-03 3.484939516e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.447619583e-04 6.174104815e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.107502732e-04 1.811379903e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.525191337e-04 8.326216815e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.991058241e-04 7.664577421e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.283475478e-04 6.654611289e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.301435675e-04 5.046254626e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 5.300276103e-05 2.911821524e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 5.747185265e-05 2.069932273e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 2.034722845e-05 1.796844258e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.143725422e-05 1.708617806e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.843338883e-05 1.600036341e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.379422229e-05 1.430262554e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.553381972e-05 1.293472910e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.317173817e-05 1.262534422e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.175644327e-05 1.252466313e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.477319312e-05 1.233129278e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 6.779141172e-06 1.186980793e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 9.169780609e-06 1.163820652e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 5.457497938e-06 1.148218956e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 6.383583052e-06 1.141340222e-09
              Solving for output 1 - done. Time (sec):  1.4260054
           Solving nonlinear problem (n=3375) - done. Time (sec):  2.7640395
        Solving for degrees of freedom - done. Time (sec):  2.8924060
     Training - done. Time (sec):  3.0869002
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
     Predicting - done. Time (sec):  0.0002041
     
     Prediction time/pt. (sec) :  0.0000020
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009961
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009999
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009973
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009973
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009966
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011568
     
     Prediction time/pt. (sec) :  0.0000116
     
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
           Computing dof2coeff - done. Time (sec):  0.0216372
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.1752834
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0524311
        Pre-computing matrices - done. Time (sec):  0.2493517
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.016131802e+05 2.067144457e+09
              Solving for output 0 - done. Time (sec):  0.1375966
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.369372950e-03 1.319294240e-07
              Solving for output 1 - done. Time (sec):  0.1383369
           Solving initial startup problem (n=2744) - done. Time (sec):  0.2759335
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.016131802e+05 2.067144457e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 3.050781756e+04 4.203956651e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 1.775862361e+04 3.531505777e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.907468303e+04 3.503624870e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 9.655279779e+03 3.373162250e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 5.062694953e+03 3.327169516e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 6.554180854e+03 3.320431226e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.564269504e+03 3.312605993e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.167873543e+03 3.307079148e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.646694990e+03 3.304613135e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.867577430e+03 3.303500055e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 1.210054705e+03 3.302115168e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.544261609e+03 3.301324967e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 7.705768299e+02 3.300091293e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 8.517446766e+02 3.298986759e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.218281099e+02 3.298287782e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.432443808e+02 3.298173834e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 4.866240749e+02 3.298109384e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 5.777224001e+02 3.298073509e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 5.610269680e+02 3.298050007e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 3.529550521e+02 3.297994272e+08
              Solving for output 0 - done. Time (sec):  2.7543049
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.369372950e-03 1.319294240e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.441448381e-04 9.432333502e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.408829469e-04 7.831583554e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.881191379e-04 5.994974878e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 9.418174431e-05 4.285209848e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 9.696691230e-05 4.048149608e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 7.135640903e-05 3.725722643e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.563442180e-05 3.356718276e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 3.630648064e-05 3.203702923e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 3.786109878e-05 3.125756795e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 1.922251659e-05 3.063894004e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.548664540e-05 3.045559977e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 1.733387307e-05 3.034590139e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.766402961e-05 3.020924552e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 3.007220113e-05 2.989135543e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 8.798737329e-06 2.949646539e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 8.226312769e-06 2.933372502e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.001142480e-05 2.929711794e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 8.404076354e-06 2.927377118e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 7.835024335e-06 2.925044524e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.383915602e-05 2.922103115e-09
              Solving for output 1 - done. Time (sec):  2.9016311
           Solving nonlinear problem (n=2744) - done. Time (sec):  5.6559360
        Solving for degrees of freedom - done. Time (sec):  5.9318695
     Training - done. Time (sec):  6.1812212
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0024099
     
     Prediction time/pt. (sec) :  0.0000241
     
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
     Predicting - done. Time (sec):  0.0084071
     
     Prediction time/pt. (sec) :  0.0000841
     
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
     Predicting - done. Time (sec):  0.0023456
     
     Prediction time/pt. (sec) :  0.0000235
     
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
     Predicting - done. Time (sec):  0.0024068
     
     Prediction time/pt. (sec) :  0.0000241
     
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
     Predicting - done. Time (sec):  0.0024052
     
     Prediction time/pt. (sec) :  0.0000241
     
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
     Predicting - done. Time (sec):  0.0024068
     
     Prediction time/pt. (sec) :  0.0000241
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
