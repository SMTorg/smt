RANS CRM wing 2-D data set
==========================

.. code-block:: python

  import numpy as np
  
  
  raw = np.array([
      [2.000000000000000000e+00 ,  4.500000000000000111e-01 ,  1.536799999999999972e-02 ,  3.674239999999999728e-01 ,  5.592279999999999474e-01 , -1.258039999999999992e-01 , -1.248699999999999984e-02],
      [3.500000000000000000e+00 ,  4.500000000000000111e-01 ,  1.985100000000000059e-02 ,  4.904470000000000218e-01 ,  7.574600000000000222e-01 , -1.615260000000000029e-01 ,  8.987000000000000197e-03],
      [5.000000000000000000e+00 ,  4.500000000000000111e-01 ,  2.571000000000000021e-02 ,  6.109189999999999898e-01 ,  9.497949999999999449e-01 , -1.954619999999999969e-01 ,  4.090900000000000092e-02],
      [6.500000000000000000e+00 ,  4.500000000000000111e-01 ,  3.304200000000000192e-02 ,  7.266120000000000356e-01 ,  1.131138999999999895e+00 , -2.255890000000000117e-01 ,  8.185399999999999621e-02],
      [8.000000000000000000e+00 ,  4.500000000000000111e-01 ,  4.318999999999999923e-02 ,  8.247250000000000414e-01 ,  1.271487000000000034e+00 , -2.397040000000000004e-01 ,  1.217659999999999992e-01],
      [0.000000000000000000e+00 ,  5.799999999999999600e-01 ,  1.136200000000000057e-02 ,  2.048760000000000026e-01 ,  2.950280000000000125e-01 , -7.882100000000000217e-02 , -2.280099999999999835e-02],
      [1.500000000000000000e+00 ,  5.799999999999999600e-01 ,  1.426000000000000011e-02 ,  3.375619999999999732e-01 ,  5.114130000000000065e-01 , -1.189420000000000061e-01 , -1.588200000000000028e-02],
      [3.000000000000000000e+00 ,  5.799999999999999600e-01 ,  1.866400000000000003e-02 ,  4.687450000000000228e-01 ,  7.240400000000000169e-01 , -1.577669999999999906e-01 ,  3.099999999999999891e-03],
      [4.500000000000000000e+00 ,  5.799999999999999600e-01 ,  2.461999999999999952e-02 ,  5.976639999999999731e-01 ,  9.311709999999999710e-01 , -1.944160000000000055e-01 ,  3.357500000000000068e-02],
      [6.000000000000000000e+00 ,  5.799999999999999600e-01 ,  3.280700000000000283e-02 ,  7.142249999999999988e-01 ,  1.111707999999999918e+00 , -2.205870000000000053e-01 ,  7.151699999999999724e-02],
      [0.000000000000000000e+00 ,  6.800000000000000488e-01 ,  1.138800000000000055e-02 ,  2.099310000000000065e-01 ,  3.032230000000000203e-01 , -8.187899999999999345e-02 , -2.172699999999999979e-02],
      [1.500000000000000000e+00 ,  6.800000000000000488e-01 ,  1.458699999999999927e-02 ,  3.518569999999999753e-01 ,  5.356630000000000003e-01 , -1.257649999999999879e-01 , -1.444800000000000077e-02],
      [3.000000000000000000e+00 ,  6.800000000000000488e-01 ,  1.952800000000000022e-02 ,  4.924879999999999813e-01 ,  7.644769999999999621e-01 , -1.678040000000000087e-01 ,  6.023999999999999841e-03],
      [4.500000000000000000e+00 ,  6.800000000000000488e-01 ,  2.666699999999999973e-02 ,  6.270339999999999803e-01 ,  9.801630000000000065e-01 , -2.035240000000000105e-01 ,  3.810000000000000192e-02],
      [6.000000000000000000e+00 ,  6.800000000000000488e-01 ,  3.891800000000000120e-02 ,  7.172730000000000494e-01 ,  1.097855999999999943e+00 , -2.014620000000000022e-01 ,  6.640000000000000069e-02],
      [0.000000000000000000e+00 ,  7.500000000000000000e-01 ,  1.150699999999999987e-02 ,  2.149069999999999869e-01 ,  3.115740000000000176e-01 , -8.498999999999999611e-02 , -2.057700000000000154e-02],
      [1.250000000000000000e+00 ,  7.500000000000000000e-01 ,  1.432600000000000019e-02 ,  3.415969999999999840e-01 ,  5.199390000000000400e-01 , -1.251009999999999900e-01 , -1.515400000000000080e-02],
      [2.500000000000000000e+00 ,  7.500000000000000000e-01 ,  1.856000000000000011e-02 ,  4.677589999999999804e-01 ,  7.262499999999999512e-01 , -1.635169999999999957e-01 ,  3.989999999999999949e-04],
      [3.750000000000000000e+00 ,  7.500000000000000000e-01 ,  2.472399999999999945e-02 ,  5.911459999999999493e-01 ,  9.254930000000000101e-01 , -1.966150000000000120e-01 ,  2.524900000000000061e-02],
      [5.000000000000000000e+00 ,  7.500000000000000000e-01 ,  3.506800000000000195e-02 ,  7.047809999999999908e-01 ,  1.097736000000000045e+00 , -2.143069999999999975e-01 ,  5.321300000000000335e-02],
      [0.000000000000000000e+00 ,  8.000000000000000444e-01 ,  1.168499999999999921e-02 ,  2.196390000000000009e-01 ,  3.197160000000000002e-01 , -8.798200000000000465e-02 , -1.926999999999999894e-02],
      [1.250000000000000000e+00 ,  8.000000000000000444e-01 ,  1.481599999999999931e-02 ,  3.553939999999999877e-01 ,  5.435950000000000504e-01 , -1.317419999999999980e-01 , -1.345599999999999921e-02],
      [2.500000000000000000e+00 ,  8.000000000000000444e-01 ,  1.968999999999999917e-02 ,  4.918299999999999894e-01 ,  7.669930000000000359e-01 , -1.728079999999999894e-01 ,  3.756999999999999923e-03],
      [3.750000000000000000e+00 ,  8.000000000000000444e-01 ,  2.785599999999999882e-02 ,  6.324319999999999942e-01 ,  9.919249999999999456e-01 , -2.077100000000000057e-01 ,  3.159800000000000109e-02],
      [5.000000000000000000e+00 ,  8.000000000000000444e-01 ,  4.394300000000000289e-02 ,  7.650689999999999991e-01 ,  1.188355999999999968e+00 , -2.332680000000000031e-01 ,  5.645000000000000018e-02],
      [0.000000000000000000e+00 ,  8.299999999999999600e-01 ,  1.186100000000000002e-02 ,  2.232899999999999885e-01 ,  3.261100000000000110e-01 , -9.028400000000000314e-02 , -1.806500000000000120e-02],
      [1.000000000000000000e+00 ,  8.299999999999999600e-01 ,  1.444900000000000004e-02 ,  3.383419999999999761e-01 ,  5.161710000000000464e-01 , -1.279530000000000112e-01 , -1.402400000000000001e-02],
      [2.000000000000000000e+00 ,  8.299999999999999600e-01 ,  1.836799999999999891e-02 ,  4.554270000000000262e-01 ,  7.082190000000000429e-01 , -1.642339999999999911e-01 , -1.793000000000000106e-03],
      [3.000000000000000000e+00 ,  8.299999999999999600e-01 ,  2.466899999999999996e-02 ,  5.798410000000000508e-01 ,  9.088819999999999677e-01 , -2.004589999999999983e-01 ,  1.892900000000000138e-02],
      [4.000000000000000000e+00 ,  8.299999999999999600e-01 ,  3.700400000000000217e-02 ,  7.012720000000000065e-01 ,  1.097366000000000064e+00 , -2.362420000000000075e-01 ,  3.750699999999999867e-02],
      [0.000000000000000000e+00 ,  8.599999999999999867e-01 ,  1.224300000000000041e-02 ,  2.278100000000000125e-01 ,  3.342720000000000136e-01 , -9.307600000000000595e-02 , -1.608400000000000107e-02],
      [1.000000000000000000e+00 ,  8.599999999999999867e-01 ,  1.540700000000000056e-02 ,  3.551839999999999997e-01 ,  5.433130000000000459e-01 , -1.364730000000000110e-01 , -1.162200000000000039e-02],
      [2.000000000000000000e+00 ,  8.599999999999999867e-01 ,  2.122699999999999934e-02 ,  4.854620000000000046e-01 ,  7.552919999999999634e-01 , -1.817850000000000021e-01 ,  1.070999999999999903e-03],
      [3.000000000000000000e+00 ,  8.599999999999999867e-01 ,  3.178899999999999781e-02 ,  6.081849999999999756e-01 ,  9.510380000000000500e-01 , -2.252020000000000133e-01 ,  1.540799999999999982e-02],
      [4.000000000000000000e+00 ,  8.599999999999999867e-01 ,  4.744199999999999806e-02 ,  6.846989999999999466e-01 ,  1.042564000000000046e+00 , -2.333600000000000119e-01 ,  2.035400000000000056e-02],
  ])
  
  
  def get_rans_crm_wing():
      # data structure:
      # alpha, mach, cd, cl, cmx, cmy, cmz
  
      deg2rad = np.pi / 180.
  
      xt = np.array(raw[:, 0:2])
      yt = np.array(raw[:, 2:4])
      xlimits = np.array([
          [-3., 10.],
          [0.4, 0.90],
      ])
  
      xt[:, 0] *= deg2rad
      xlimits[0, :] *= deg2rad
  
      return xt, yt, xlimits
  
  
  def plot_rans_crm_wing(xt, yt, limits, interp):
      import numpy as np
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
  
      rad2deg = 180. / np.pi
  
      num = 500
      num_a = 50
      num_M = 50
  
      x = np.zeros((num, 2))
      colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
  
      nrow = 3
      ncol = 2
  
      plt.close()
      plt.figure(figsize=(15, 15))
  
      # -----------------------------------------------------------------------------
  
      mach_numbers = [0.45, 0.68, 0.80, 0.86]
      legend_entries = []
  
      alpha_sweep = np.linspace(0., 8., num)
  
      for ind, mach in enumerate(mach_numbers):
          x[:, 0] = alpha_sweep / rad2deg
          x[:, 1] = mach
          CD = interp.predict_values(x)[:, 0]
          CL = interp.predict_values(x)[:, 1]
  
          plt.subplot(nrow, ncol, 1)
  
          mask = np.abs(xt[:, 1] - mach) < 1e-10
          plt.plot(xt[mask, 0] * rad2deg, yt[mask, 0], 'o' + colors[ind])
          plt.plot(alpha_sweep, CD, colors[ind])
  
          plt.subplot(nrow, ncol, 2)
  
          mask = np.abs(xt[:, 1] - mach) < 1e-10
          plt.plot(xt[mask, 0] * rad2deg, yt[mask, 1], 'o' + colors[ind])
          plt.plot(alpha_sweep, CL, colors[ind])
  
          legend_entries.append('M={}'.format(mach))
          legend_entries.append('exact')
  
      plt.subplot(nrow, ncol, 1)
      plt.xlabel('alpha (deg)')
      plt.ylabel('CD')
      plt.legend(legend_entries)
  
      plt.subplot(nrow, ncol, 2)
      plt.xlabel('alpha (deg)')
      plt.ylabel('CL')
      plt.legend(legend_entries)
  
      # -----------------------------------------------------------------------------
  
      alphas = [2., 4., 6.]
      legend_entries = []
  
      mach_sweep = np.linspace(0.45, 0.86, num)
  
      for ind, alpha in enumerate(alphas):
          x[:, 0] = alpha / rad2deg
          x[:, 1] = mach_sweep
          CD = interp.predict_values(x)[:, 0]
          CL = interp.predict_values(x)[:, 1]
  
          plt.subplot(nrow, ncol, 3)
          plt.plot(mach_sweep, CD, colors[ind])
  
          plt.subplot(nrow, ncol, 4)
          plt.plot(mach_sweep, CL, colors[ind])
  
          legend_entries.append('alpha={}'.format(alpha))
  
      plt.subplot(nrow, ncol, 3)
      plt.xlabel('Mach number')
      plt.ylabel('CD')
      plt.legend(legend_entries)
  
      plt.subplot(nrow, ncol, 4)
      plt.xlabel('Mach number')
      plt.ylabel('CL')
      plt.legend(legend_entries)
  
      # -----------------------------------------------------------------------------
  
      x = np.zeros((num_a, num_M, 2))
      x[:, :, 0] = np.outer(np.linspace(0., 8., num_a), np.ones(num_M)) / rad2deg
      x[:, :, 1] = np.outer(np.ones(num_a), np.linspace(0.45, 0.86, num_M))
      CD = interp.predict_values(x.reshape((num_a * num_M, 2)))[:, 0].reshape((num_a, num_M))
      CL = interp.predict_values(x.reshape((num_a * num_M, 2)))[:, 1].reshape((num_a, num_M))
  
      plt.subplot(nrow, ncol, 5)
      plt.plot(xt[:, 1], xt[:, 0] * rad2deg, 'o')
      plt.contour(x[:, :, 1], x[:, :, 0] * rad2deg, CD, 20)
      plt.pcolormesh(x[:, :, 1], x[:, :, 0] * rad2deg, CD, cmap = plt.get_cmap('rainbow'))
      plt.xlabel('Mach number')
      plt.ylabel('alpha (deg)')
      plt.title('CD')
      plt.colorbar()
  
      plt.subplot(nrow, ncol, 6)
      plt.plot(xt[:, 1], xt[:, 0] * rad2deg, 'o')
      plt.contour(x[:, :, 1], x[:, :, 0] * rad2deg, CL, 20)
      plt.pcolormesh(x[:, :, 1], x[:, :, 0] * rad2deg, CL, cmap = plt.get_cmap('rainbow'))
      plt.xlabel('Mach number')
      plt.ylabel('alpha (deg)')
      plt.title('CL')
      plt.colorbar()
  
      plt.show()
  

RMTB
----

.. code-block:: python

  from smt.surrogate_models import RMTB
  from smt.examples.rans_crm_wing.rans_crm_wing import get_rans_crm_wing, plot_rans_crm_wing
  
  xt, yt, xlimits = get_rans_crm_wing()
  
  interp = RMTB(num_ctrl_pts=20, xlimits=xlimits, nonlinear_maxiter=100, energy_weight=1e-12)
  interp.set_training_values(xt, yt)
  interp.train()
  
  plot_rans_crm_wing(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTB
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 35
     
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
           Solving initial startup problem (n=400) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.429150220e-02 1.114942861e-02
                 Iteration (num., iy, grad. norm, func.) :   0   0 4.818616596e-08 1.793075513e-10
              Solving for output 0 - done. Time (sec):  0.0159998
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.955493282e+00 4.799845498e+00
                 Iteration (num., iy, grad. norm, func.) :   0   1 2.419601604e-07 4.567655233e-08
              Solving for output 1 - done. Time (sec):  0.0000000
           Solving initial startup problem (n=400) - done. Time (sec):  0.0159998
           Solving nonlinear problem (n=400) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 6.652245258e-09 1.793037495e-10
                 Iteration (num., iy, grad. norm, func.) :   0   0 5.849160490e-09 1.703962484e-10
                 Iteration (num., iy, grad. norm, func.) :   1   0 3.036212379e-08 1.036013113e-10
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.128756868e-08 2.514073559e-11
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.619621088e-09 1.059551607e-11
                 Iteration (num., iy, grad. norm, func.) :   4   0 1.835273882e-09 8.860268038e-12
                 Iteration (num., iy, grad. norm, func.) :   5   0 5.183428875e-10 7.289436313e-12
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.424249810e-10 6.502543214e-12
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.490980639e-11 6.259868254e-12
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.986725150e-11 6.257846523e-12
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.117468737e-11 6.257275387e-12
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.902259181e-12 6.257250516e-12
                 Iteration (num., iy, grad. norm, func.) :  11   0 5.389791968e-12 6.256231549e-12
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.241489422e-13 6.255724775e-12
              Solving for output 0 - done. Time (sec):  0.1240001
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 9.729426342e-08 4.567642734e-08
                 Iteration (num., iy, grad. norm, func.) :   0   1 9.338189719e-08 4.538213544e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.907368537e-06 3.252405829e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 8.618936732e-07 4.664691176e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 3.250438862e-07 2.346732225e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 3.055490309e-07 2.193001360e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 9.325493881e-08 7.021223910e-10
                 Iteration (num., iy, grad. norm, func.) :   6   1 2.701748068e-08 5.036964840e-10
                 Iteration (num., iy, grad. norm, func.) :   7   1 8.837235724e-09 4.770629100e-10
                 Iteration (num., iy, grad. norm, func.) :   8   1 1.487749322e-08 3.759155971e-10
                 Iteration (num., iy, grad. norm, func.) :   9   1 4.377212088e-09 2.946929497e-10
                 Iteration (num., iy, grad. norm, func.) :  10   1 1.270985227e-09 2.731643763e-10
                 Iteration (num., iy, grad. norm, func.) :  11   1 7.673505063e-10 2.730152868e-10
                 Iteration (num., iy, grad. norm, func.) :  12   1 4.537707295e-10 2.726301672e-10
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.521325905e-10 2.721125872e-10
                 Iteration (num., iy, grad. norm, func.) :  14   1 5.311966392e-11 2.715071295e-10
                 Iteration (num., iy, grad. norm, func.) :  15   1 8.545116438e-11 2.714798653e-10
                 Iteration (num., iy, grad. norm, func.) :  16   1 5.269447073e-11 2.714525867e-10
                 Iteration (num., iy, grad. norm, func.) :  17   1 3.327746829e-11 2.714083863e-10
                 Iteration (num., iy, grad. norm, func.) :  18   1 2.522963973e-11 2.713733200e-10
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.340324923e-11 2.713595357e-10
                 Iteration (num., iy, grad. norm, func.) :  20   1 4.601694243e-11 2.713584654e-10
                 Iteration (num., iy, grad. norm, func.) :  21   1 7.023710586e-12 2.713523423e-10
                 Iteration (num., iy, grad. norm, func.) :  22   1 1.130153674e-11 2.713508984e-10
                 Iteration (num., iy, grad. norm, func.) :  23   1 1.085024722e-11 2.713478550e-10
                 Iteration (num., iy, grad. norm, func.) :  24   1 6.505739219e-12 2.713457809e-10
                 Iteration (num., iy, grad. norm, func.) :  25   1 1.870222792e-12 2.713454496e-10
                 Iteration (num., iy, grad. norm, func.) :  26   1 3.213975858e-12 2.713453968e-10
                 Iteration (num., iy, grad. norm, func.) :  27   1 1.433125473e-12 2.713451448e-10
                 Iteration (num., iy, grad. norm, func.) :  28   1 1.081435426e-12 2.713449905e-10
                 Iteration (num., iy, grad. norm, func.) :  29   1 9.761069515e-13 2.713449654e-10
              Solving for output 1 - done. Time (sec):  0.2660000
           Solving nonlinear problem (n=400) - done. Time (sec):  0.3900001
        Solving for degrees of freedom - done. Time (sec):  0.4059999
     Training - done. Time (sec):  0.4059999
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: rans_crm_wing.png
  :scale: 60 %
  :align: center

RMTC
----

.. code-block:: python

  from smt.surrogate_models import RMTC
  from smt.examples.rans_crm_wing.rans_crm_wing import get_rans_crm_wing, plot_rans_crm_wing
  
  xt, yt, xlimits = get_rans_crm_wing()
  
  interp = RMTC(num_elements=20, xlimits=xlimits, nonlinear_maxiter=100, energy_weight=1e-10)
  interp.set_training_values(xt, yt)
  interp.train()
  
  plot_rans_crm_wing(xt, yt, xlimits, interp)
  
::

  ___________________________________________________________________________
     
                                     RMTC
  ___________________________________________________________________________
     
   Problem size
     
        # training points.        : 35
     
  ___________________________________________________________________________
     
   Training
     
     Training ...
        Pre-computing matrices ...
           Computing dof2coeff ...
           Computing dof2coeff - done. Time (sec):  0.0160000
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0000000
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0149999
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0000000
        Pre-computing matrices - done. Time (sec):  0.0309999
        Solving for degrees of freedom ...
           Solving initial startup problem (n=1764) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.279175539e-01 1.114942861e-02
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.404552577e-05 2.114195757e-08
              Solving for output 0 - done. Time (sec):  0.0160000
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 2.653045755e+00 4.799845498e+00
                 Iteration (num., iy, grad. norm, func.) :   0   1 7.810833533e-05 6.499602611e-06
              Solving for output 1 - done. Time (sec):  0.0310001
           Solving initial startup problem (n=1764) - done. Time (sec):  0.0470002
           Solving nonlinear problem (n=1764) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 8.159618595e-07 2.098157518e-08
                 Iteration (num., iy, grad. norm, func.) :   0   0 8.842962484e-07 1.664368008e-08
                 Iteration (num., iy, grad. norm, func.) :   1   0 3.428412618e-07 3.193178101e-09
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.137256306e-07 1.029343891e-09
                 Iteration (num., iy, grad. norm, func.) :   3   0 6.114924433e-08 5.279633103e-10
                 Iteration (num., iy, grad. norm, func.) :   4   0 3.579156034e-08 4.131646637e-10
                 Iteration (num., iy, grad. norm, func.) :   5   0 2.313281559e-08 3.786839299e-10
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.054135656e-08 3.748189364e-10
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.043652614e-08 3.736036009e-10
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.338845451e-08 3.613571303e-10
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.605254098e-08 3.414371204e-10
                 Iteration (num., iy, grad. norm, func.) :  10   0 6.539042953e-09 3.053774577e-10
                 Iteration (num., iy, grad. norm, func.) :  11   0 2.163189116e-09 2.891970198e-10
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.596319877e-09 2.876375738e-10
                 Iteration (num., iy, grad. norm, func.) :  13   0 2.068004803e-09 2.875687684e-10
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.397412209e-09 2.873891421e-10
                 Iteration (num., iy, grad. norm, func.) :  15   0 2.941332101e-09 2.872203471e-10
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.013974342e-10 2.867659335e-10
                 Iteration (num., iy, grad. norm, func.) :  17   0 2.465583872e-10 2.867613167e-10
                 Iteration (num., iy, grad. norm, func.) :  18   0 1.151862574e-09 2.867408845e-10
                 Iteration (num., iy, grad. norm, func.) :  19   0 5.189289071e-10 2.866836883e-10
                 Iteration (num., iy, grad. norm, func.) :  20   0 1.115919722e-09 2.866438936e-10
                 Iteration (num., iy, grad. norm, func.) :  21   0 2.643815200e-10 2.865594364e-10
                 Iteration (num., iy, grad. norm, func.) :  22   0 2.171583178e-10 2.865503472e-10
                 Iteration (num., iy, grad. norm, func.) :  23   0 4.054357915e-10 2.865437412e-10
                 Iteration (num., iy, grad. norm, func.) :  24   0 3.340051930e-10 2.865364777e-10
                 Iteration (num., iy, grad. norm, func.) :  25   0 5.594290245e-10 2.865327721e-10
                 Iteration (num., iy, grad. norm, func.) :  26   0 2.405096122e-10 2.865238620e-10
                 Iteration (num., iy, grad. norm, func.) :  27   0 4.112131222e-10 2.865130212e-10
                 Iteration (num., iy, grad. norm, func.) :  28   0 1.188601847e-10 2.865016209e-10
                 Iteration (num., iy, grad. norm, func.) :  29   0 9.281008872e-11 2.865002751e-10
                 Iteration (num., iy, grad. norm, func.) :  30   0 1.438087912e-10 2.864999917e-10
                 Iteration (num., iy, grad. norm, func.) :  31   0 1.652880566e-10 2.864999693e-10
                 Iteration (num., iy, grad. norm, func.) :  32   0 1.309192831e-10 2.864996387e-10
                 Iteration (num., iy, grad. norm, func.) :  33   0 1.957008110e-10 2.864979521e-10
                 Iteration (num., iy, grad. norm, func.) :  34   0 6.490995009e-11 2.864956204e-10
                 Iteration (num., iy, grad. norm, func.) :  35   0 7.758276681e-11 2.864947482e-10
                 Iteration (num., iy, grad. norm, func.) :  36   0 6.448062147e-11 2.864944339e-10
                 Iteration (num., iy, grad. norm, func.) :  37   0 8.225946136e-11 2.864934244e-10
                 Iteration (num., iy, grad. norm, func.) :  38   0 2.184439745e-11 2.864927451e-10
                 Iteration (num., iy, grad. norm, func.) :  39   0 2.184438609e-11 2.864927451e-10
                 Iteration (num., iy, grad. norm, func.) :  40   0 2.306623456e-11 2.864927216e-10
                 Iteration (num., iy, grad. norm, func.) :  41   0 2.681294466e-11 2.864926518e-10
                 Iteration (num., iy, grad. norm, func.) :  42   0 1.916671429e-11 2.864926174e-10
                 Iteration (num., iy, grad. norm, func.) :  43   0 3.012516271e-11 2.864926146e-10
                 Iteration (num., iy, grad. norm, func.) :  44   0 2.000669157e-11 2.864925825e-10
                 Iteration (num., iy, grad. norm, func.) :  45   0 1.558681340e-11 2.864925295e-10
                 Iteration (num., iy, grad. norm, func.) :  46   0 1.287290566e-11 2.864924979e-10
                 Iteration (num., iy, grad. norm, func.) :  47   0 1.482058915e-11 2.864924775e-10
                 Iteration (num., iy, grad. norm, func.) :  48   0 1.617905135e-11 2.864924581e-10
                 Iteration (num., iy, grad. norm, func.) :  49   0 6.847528984e-12 2.864924403e-10
                 Iteration (num., iy, grad. norm, func.) :  50   0 5.742660930e-12 2.864924378e-10
                 Iteration (num., iy, grad. norm, func.) :  51   0 7.864800854e-12 2.864924358e-10
                 Iteration (num., iy, grad. norm, func.) :  52   0 7.248265633e-12 2.864924327e-10
                 Iteration (num., iy, grad. norm, func.) :  53   0 6.866308485e-12 2.864924290e-10
                 Iteration (num., iy, grad. norm, func.) :  54   0 4.614283514e-12 2.864924249e-10
                 Iteration (num., iy, grad. norm, func.) :  55   0 4.822100054e-12 2.864924235e-10
                 Iteration (num., iy, grad. norm, func.) :  56   0 3.361060410e-12 2.864924221e-10
                 Iteration (num., iy, grad. norm, func.) :  57   0 5.211096502e-12 2.864924195e-10
                 Iteration (num., iy, grad. norm, func.) :  58   0 1.407416477e-12 2.864924169e-10
                 Iteration (num., iy, grad. norm, func.) :  59   0 1.241396492e-12 2.864924169e-10
                 Iteration (num., iy, grad. norm, func.) :  60   0 1.347449604e-12 2.864924168e-10
                 Iteration (num., iy, grad. norm, func.) :  61   0 2.385015350e-12 2.864924166e-10
                 Iteration (num., iy, grad. norm, func.) :  62   0 1.591143631e-12 2.864924164e-10
                 Iteration (num., iy, grad. norm, func.) :  63   0 1.445870182e-12 2.864924162e-10
                 Iteration (num., iy, grad. norm, func.) :  64   0 1.673901249e-12 2.864924161e-10
                 Iteration (num., iy, grad. norm, func.) :  65   0 8.889567845e-13 2.864924159e-10
              Solving for output 0 - done. Time (sec):  1.4510000
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.428804416e-05 6.494698255e-06
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.428876115e-05 6.248020336e-06
                 Iteration (num., iy, grad. norm, func.) :   1   1 1.438242492e-05 8.010340697e-07
                 Iteration (num., iy, grad. norm, func.) :   2   1 1.870801960e-05 3.733657029e-07
                 Iteration (num., iy, grad. norm, func.) :   3   1 5.702798559e-06 1.294881398e-07
                 Iteration (num., iy, grad. norm, func.) :   4   1 4.680032521e-06 1.013704437e-07
                 Iteration (num., iy, grad. norm, func.) :   5   1 1.464582949e-06 3.735233064e-08
                 Iteration (num., iy, grad. norm, func.) :   6   1 9.998204056e-07 3.019495176e-08
                 Iteration (num., iy, grad. norm, func.) :   7   1 7.700187788e-07 2.961836224e-08
                 Iteration (num., iy, grad. norm, func.) :   8   1 5.121865284e-07 2.868916239e-08
                 Iteration (num., iy, grad. norm, func.) :   9   1 1.883963662e-07 2.307291277e-08
                 Iteration (num., iy, grad. norm, func.) :  10   1 1.207055533e-07 1.782199864e-08
                 Iteration (num., iy, grad. norm, func.) :  11   1 4.100710238e-08 1.507123663e-08
                 Iteration (num., iy, grad. norm, func.) :  12   1 4.383992474e-08 1.487056981e-08
                 Iteration (num., iy, grad. norm, func.) :  13   1 4.383992474e-08 1.487056981e-08
                 Iteration (num., iy, grad. norm, func.) :  14   1 4.383992473e-08 1.487056981e-08
                 Iteration (num., iy, grad. norm, func.) :  15   1 3.695798112e-08 1.479137977e-08
                 Iteration (num., iy, grad. norm, func.) :  16   1 2.218860766e-08 1.466183152e-08
                 Iteration (num., iy, grad. norm, func.) :  17   1 2.356901504e-08 1.458210337e-08
                 Iteration (num., iy, grad. norm, func.) :  18   1 1.012035305e-08 1.451356650e-08
                 Iteration (num., iy, grad. norm, func.) :  19   1 8.658292942e-09 1.449090952e-08
                 Iteration (num., iy, grad. norm, func.) :  20   1 7.321699140e-09 1.448999050e-08
                 Iteration (num., iy, grad. norm, func.) :  21   1 8.998014508e-09 1.448883538e-08
                 Iteration (num., iy, grad. norm, func.) :  22   1 7.196344538e-09 1.448284111e-08
                 Iteration (num., iy, grad. norm, func.) :  23   1 8.004107725e-09 1.447713861e-08
                 Iteration (num., iy, grad. norm, func.) :  24   1 3.775992485e-09 1.447211098e-08
                 Iteration (num., iy, grad. norm, func.) :  25   1 5.256130544e-09 1.447029542e-08
                 Iteration (num., iy, grad. norm, func.) :  26   1 3.046334424e-09 1.446899185e-08
                 Iteration (num., iy, grad. norm, func.) :  27   1 4.628458755e-09 1.446806004e-08
                 Iteration (num., iy, grad. norm, func.) :  28   1 1.800586012e-09 1.446658425e-08
                 Iteration (num., iy, grad. norm, func.) :  29   1 2.816288485e-09 1.446636176e-08
                 Iteration (num., iy, grad. norm, func.) :  30   1 1.714311764e-09 1.446577552e-08
                 Iteration (num., iy, grad. norm, func.) :  31   1 2.477603916e-09 1.446511332e-08
                 Iteration (num., iy, grad. norm, func.) :  32   1 1.095716128e-09 1.446448248e-08
                 Iteration (num., iy, grad. norm, func.) :  33   1 1.577273668e-09 1.446417087e-08
                 Iteration (num., iy, grad. norm, func.) :  34   1 7.674105229e-10 1.446404735e-08
                 Iteration (num., iy, grad. norm, func.) :  35   1 1.031734597e-09 1.446399455e-08
                 Iteration (num., iy, grad. norm, func.) :  36   1 6.436454480e-10 1.446388563e-08
                 Iteration (num., iy, grad. norm, func.) :  37   1 8.573647141e-10 1.446376665e-08
                 Iteration (num., iy, grad. norm, func.) :  38   1 4.317645005e-10 1.446369499e-08
                 Iteration (num., iy, grad. norm, func.) :  39   1 6.075669017e-10 1.446367791e-08
                 Iteration (num., iy, grad. norm, func.) :  40   1 7.005375748e-10 1.446365009e-08
                 Iteration (num., iy, grad. norm, func.) :  41   1 3.384153753e-10 1.446361354e-08
                 Iteration (num., iy, grad. norm, func.) :  42   1 2.553003306e-10 1.446360609e-08
                 Iteration (num., iy, grad. norm, func.) :  43   1 3.672154009e-10 1.446359304e-08
                 Iteration (num., iy, grad. norm, func.) :  44   1 2.386315492e-10 1.446358101e-08
                 Iteration (num., iy, grad. norm, func.) :  45   1 2.418051100e-10 1.446357537e-08
                 Iteration (num., iy, grad. norm, func.) :  46   1 1.355482999e-10 1.446357234e-08
                 Iteration (num., iy, grad. norm, func.) :  47   1 1.107451745e-10 1.446357100e-08
                 Iteration (num., iy, grad. norm, func.) :  48   1 1.423661156e-10 1.446356879e-08
                 Iteration (num., iy, grad. norm, func.) :  49   1 1.338390031e-10 1.446356522e-08
                 Iteration (num., iy, grad. norm, func.) :  50   1 8.427496807e-11 1.446356292e-08
                 Iteration (num., iy, grad. norm, func.) :  51   1 9.926047993e-11 1.446356211e-08
                 Iteration (num., iy, grad. norm, func.) :  52   1 7.805966344e-11 1.446356154e-08
                 Iteration (num., iy, grad. norm, func.) :  53   1 7.365319709e-11 1.446356093e-08
                 Iteration (num., iy, grad. norm, func.) :  54   1 5.550710608e-11 1.446356074e-08
                 Iteration (num., iy, grad. norm, func.) :  55   1 7.531667871e-11 1.446356033e-08
                 Iteration (num., iy, grad. norm, func.) :  56   1 3.986837552e-11 1.446355981e-08
                 Iteration (num., iy, grad. norm, func.) :  57   1 3.365740451e-11 1.446355948e-08
                 Iteration (num., iy, grad. norm, func.) :  58   1 2.385208751e-11 1.446355938e-08
                 Iteration (num., iy, grad. norm, func.) :  59   1 1.921113734e-11 1.446355937e-08
                 Iteration (num., iy, grad. norm, func.) :  60   1 2.671520330e-11 1.446355936e-08
                 Iteration (num., iy, grad. norm, func.) :  61   1 1.848026773e-11 1.446355928e-08
                 Iteration (num., iy, grad. norm, func.) :  62   1 1.212120177e-11 1.446355923e-08
                 Iteration (num., iy, grad. norm, func.) :  63   1 1.075741298e-11 1.446355921e-08
                 Iteration (num., iy, grad. norm, func.) :  64   1 1.464873087e-11 1.446355920e-08
                 Iteration (num., iy, grad. norm, func.) :  65   1 1.027281624e-11 1.446355919e-08
                 Iteration (num., iy, grad. norm, func.) :  66   1 1.297185646e-11 1.446355919e-08
                 Iteration (num., iy, grad. norm, func.) :  67   1 6.745480716e-12 1.446355918e-08
                 Iteration (num., iy, grad. norm, func.) :  68   1 8.190743813e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  69   1 4.113668694e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  70   1 4.945543590e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  71   1 4.172766271e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  72   1 7.743031868e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  73   1 1.875352120e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  74   1 2.028292439e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  75   1 2.326961807e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  76   1 2.905109609e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  77   1 1.930979092e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  78   1 2.873934150e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  79   1 1.154260938e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  80   1 2.172221246e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  81   1 6.196580463e-13 1.446355915e-08
              Solving for output 1 - done. Time (sec):  1.8099999
           Solving nonlinear problem (n=1764) - done. Time (sec):  3.2609999
        Solving for degrees of freedom - done. Time (sec):  3.3080001
     Training - done. Time (sec):  3.3390000
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0160000
     
     Prediction time/pt. (sec) :  0.0000320
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0000000
     
     Prediction time/pt. (sec) :  0.0000000
     
  
.. figure:: rans_crm_wing.png
  :scale: 60 %
  :align: center
