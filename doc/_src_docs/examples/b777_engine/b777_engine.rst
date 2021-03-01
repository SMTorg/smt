Boeing 777 engine data set
==========================

.. code-block:: python

  import numpy as np
  import os
  
  
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
      import numpy as np
      import matplotlib
  
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
  
      fig, axs = plt.subplots(6, 2, gridspec_kw={"hspace": 0.5}, figsize=(15, 25))
  
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

  from smt.surrogate_models import RMTB
  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  
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
           Initializing Hessian - done. Time (sec):  0.0005000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.2890003
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0104997
        Pre-computing matrices - done. Time (sec):  0.3000000
        Solving for degrees of freedom ...
           Solving initial startup problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.857178281e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.364349733e+05 7.002441710e+09
              Solving for output 0 - done. Time (sec):  0.0985003
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.711896708e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.384257034e-03 3.512467641e-07
              Solving for output 1 - done. Time (sec):  0.0939999
           Solving initial startup problem (n=3375) - done. Time (sec):  0.1925001
           Solving nonlinear problem (n=3375) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.364349733e+05 7.002441710e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.401682427e+04 1.956585489e+09
                 Iteration (num., iy, grad. norm, func.) :   1   0 4.640761309e+04 5.653768085e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 3.726949662e+04 3.860194807e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.244331543e+04 3.735217325e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 2.356309977e+04 3.232040667e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 1.896770441e+04 2.970854602e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.168979712e+04 2.643923864e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.199133401e+04 2.223771115e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 9.363877631e+03 2.013234589e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 9.544160641e+03 1.861724031e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.458916793e+03 1.762819815e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 4.152198214e+03 1.661887141e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.359804107e+03 1.619868009e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 2.678073894e+03 1.599839425e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 2.301049932e+03 1.583627245e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 3.127472449e+03 1.554361115e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 2.879195835e+03 1.516054749e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 1.583184160e+03 1.493412967e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 2.202973513e+03 1.492035778e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.397841194e+03 1.489828558e+08
              Solving for output 0 - done. Time (sec):  2.2365003
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.384257034e-03 3.512467641e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 3.575138262e-04 6.166597300e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 3.156992731e-04 1.817140551e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.070220585e-04 8.504635606e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.711558893e-04 7.824284644e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 1.147466159e-04 6.729973912e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.033293877e-04 5.063463186e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 5.272698157e-05 2.929839938e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 4.894442104e-05 2.071717930e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 2.850823295e-05 1.797321609e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.566163204e-05 1.713105879e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.728118053e-05 1.606498899e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.407731298e-05 1.439553327e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.588414550e-05 1.302254672e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 1.941516089e-05 1.258276496e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 1.159190980e-05 1.239434907e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 1.872674427e-05 1.235569556e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.169536710e-05 1.206341167e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.005666171e-05 1.172498758e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 4.240888944e-06 1.143928197e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 4.653082813e-06 1.142989811e-09
              Solving for output 1 - done. Time (sec):  2.1650000
           Solving nonlinear problem (n=3375) - done. Time (sec):  4.4015002
        Solving for degrees of freedom - done. Time (sec):  4.5940003
     Training - done. Time (sec):  4.8955002
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0014999
     
     Prediction time/pt. (sec) :  0.0000150
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0014999
     
     Prediction time/pt. (sec) :  0.0000150
     
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
     Predicting - done. Time (sec):  0.0009999
     
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
     Predicting - done. Time (sec):  0.0009999
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0014999
     
     Prediction time/pt. (sec) :  0.0000150
     
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
     Predicting - done. Time (sec):  0.0014999
     
     Prediction time/pt. (sec) :  0.0000150
     
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
     Predicting - done. Time (sec):  0.0009999
     
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
     Predicting - done. Time (sec):  0.0009999
     
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
     Predicting - done. Time (sec):  0.0009999
     
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
     Predicting - done. Time (sec):  0.0009999
     
     Prediction time/pt. (sec) :  0.0000100
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0015001
     
     Prediction time/pt. (sec) :  0.0000150
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.surrogate_models import RMTC
  from smt.examples.b777_engine.b777_engine import get_b777_engine, plot_b777_engine
  
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
           Computing dof2coeff - done. Time (sec):  0.0265002
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0005000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.1715000
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0704999
        Pre-computing matrices - done. Time (sec):  0.2690001
        Solving for degrees of freedom ...
           Solving initial startup problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 7.864862172e+07 2.642628384e+13
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.060431396e+05 2.065425781e+09
              Solving for output 0 - done. Time (sec):  0.2050002
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 8.095040141e-01 7.697335516e-04
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.282275071e-03 1.321733275e-07
              Solving for output 1 - done. Time (sec):  0.2164998
           Solving initial startup problem (n=2744) - done. Time (sec):  0.4215000
           Solving nonlinear problem (n=2744) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.060431396e+05 2.065425781e+09
                 Iteration (num., iy, grad. norm, func.) :   0   0 3.052468349e+04 4.214887369e+08
                 Iteration (num., iy, grad. norm, func.) :   1   0 1.637519741e+04 3.528333013e+08
                 Iteration (num., iy, grad. norm, func.) :   2   0 2.467565872e+04 3.502315345e+08
                 Iteration (num., iy, grad. norm, func.) :   3   0 1.240340684e+04 3.371969581e+08
                 Iteration (num., iy, grad. norm, func.) :   4   0 4.484518190e+03 3.326819382e+08
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.458664702e+03 3.320567594e+08
                 Iteration (num., iy, grad. norm, func.) :   6   0 3.702985472e+03 3.312693307e+08
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.182133942e+03 3.307167443e+08
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.621754330e+03 3.304715643e+08
                 Iteration (num., iy, grad. norm, func.) :   9   0 2.240399133e+03 3.303481442e+08
                 Iteration (num., iy, grad. norm, func.) :  10   0 8.445161587e+02 3.302027288e+08
                 Iteration (num., iy, grad. norm, func.) :  11   0 1.714003020e+03 3.301262590e+08
                 Iteration (num., iy, grad. norm, func.) :  12   0 9.880609256e+02 3.300024854e+08
                 Iteration (num., iy, grad. norm, func.) :  13   0 8.521539080e+02 3.298999867e+08
                 Iteration (num., iy, grad. norm, func.) :  14   0 4.282893933e+02 3.298348773e+08
                 Iteration (num., iy, grad. norm, func.) :  15   0 5.092320400e+02 3.298219882e+08
                 Iteration (num., iy, grad. norm, func.) :  16   0 4.692038132e+02 3.298129488e+08
                 Iteration (num., iy, grad. norm, func.) :  17   0 6.745357115e+02 3.298066225e+08
                 Iteration (num., iy, grad. norm, func.) :  18   0 4.964325515e+02 3.298021330e+08
                 Iteration (num., iy, grad. norm, func.) :  19   0 3.041372893e+02 3.297966130e+08
              Solving for output 0 - done. Time (sec):  4.1695001
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.282275071e-03 1.321733275e-07
                 Iteration (num., iy, grad. norm, func.) :   0   1 4.122910555e-04 9.569245716e-09
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.755337104e-04 7.866451391e-09
                 Iteration (num., iy, grad. norm, func.) :   2   1 2.701197307e-04 5.988632553e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 1.030664312e-04 4.277431523e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 9.296966525e-05 4.063313099e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 5.008243419e-05 3.747746015e-09
                 Iteration (num., iy, grad. norm, func.) :   6   1 5.748405190e-05 3.365231442e-09
                 Iteration (num., iy, grad. norm, func.) :   7   1 4.114360860e-05 3.201633381e-09
                 Iteration (num., iy, grad. norm, func.) :   8   1 3.967337252e-05 3.123204721e-09
                 Iteration (num., iy, grad. norm, func.) :   9   1 3.632681351e-05 3.064600592e-09
                 Iteration (num., iy, grad. norm, func.) :  10   1 2.184638932e-05 3.035599200e-09
                 Iteration (num., iy, grad. norm, func.) :  11   1 2.357255269e-05 3.034890988e-09
                 Iteration (num., iy, grad. norm, func.) :  12   1 1.803826098e-05 3.019435094e-09
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.841331088e-05 2.987531909e-09
                 Iteration (num., iy, grad. norm, func.) :  14   1 8.297424592e-06 2.942596606e-09
                 Iteration (num., iy, grad. norm, func.) :  15   1 7.815593710e-06 2.929156592e-09
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.341357979e-05 2.928335817e-09
                 Iteration (num., iy, grad. norm, func.) :  17   1 8.761911101e-06 2.927242317e-09
                 Iteration (num., iy, grad. norm, func.) :  18   1 9.400759968e-06 2.926519648e-09
                 Iteration (num., iy, grad. norm, func.) :  19   1 8.023846670e-06 2.922260441e-09
              Solving for output 1 - done. Time (sec):  4.1485000
           Solving nonlinear problem (n=2744) - done. Time (sec):  8.3180001
        Solving for degrees of freedom - done. Time (sec):  8.7395000
     Training - done. Time (sec):  9.0115001
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0035000
     
     Prediction time/pt. (sec) :  0.0000350
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030000
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0029998
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030003
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0029998
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0035000
     
     Prediction time/pt. (sec) :  0.0000350
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030003
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0024998
     
     Prediction time/pt. (sec) :  0.0000250
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030000
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0029998
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0025001
     
     Prediction time/pt. (sec) :  0.0000250
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030000
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0035000
     
     Prediction time/pt. (sec) :  0.0000350
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030000
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030000
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0029998
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0030000
     
     Prediction time/pt. (sec) :  0.0000300
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 100
     
     Predicting ...
     Predicting - done. Time (sec):  0.0029998
     
     Prediction time/pt. (sec) :  0.0000300
     
  
.. figure:: b777_engine.png
  :scale: 60 %
  :align: center
