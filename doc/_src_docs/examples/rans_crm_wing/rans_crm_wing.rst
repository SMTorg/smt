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
           Computing dof2coeff - done. Time (sec):  0.0000041
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0008872
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0119421
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0010629
        Pre-computing matrices - done. Time (sec):  0.0140262
        Solving for degrees of freedom ...
           Solving initial startup problem (n=400) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.429150220e-02 1.114942861e-02
                 Iteration (num., iy, grad. norm, func.) :   0   0 2.918130789e-08 1.793051131e-10
              Solving for output 0 - done. Time (sec):  0.0252142
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.955493282e+00 4.799845498e+00
                 Iteration (num., iy, grad. norm, func.) :   0   1 5.170087671e-07 4.567684873e-08
              Solving for output 1 - done. Time (sec):  0.0257990
           Solving initial startup problem (n=400) - done. Time (sec):  0.0511999
           Solving nonlinear problem (n=400) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 6.652554858e-09 1.793037268e-10
                 Iteration (num., iy, grad. norm, func.) :   0   0 5.849425817e-09 1.703967811e-10
                 Iteration (num., iy, grad. norm, func.) :   1   0 3.040889961e-08 1.037059796e-10
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.131130625e-08 2.516658925e-11
                 Iteration (num., iy, grad. norm, func.) :   3   0 3.676281227e-09 1.068748129e-11
                 Iteration (num., iy, grad. norm, func.) :   4   0 1.693073866e-09 8.718637681e-12
                 Iteration (num., iy, grad. norm, func.) :   5   0 4.767435676e-10 7.258287806e-12
                 Iteration (num., iy, grad. norm, func.) :   6   0 1.292737721e-10 6.494082490e-12
                 Iteration (num., iy, grad. norm, func.) :   7   0 2.129984898e-11 6.259342852e-12
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.701491451e-11 6.257526556e-12
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.055740158e-11 6.257128928e-12
                 Iteration (num., iy, grad. norm, func.) :  10   0 9.367344079e-12 6.257118464e-12
                 Iteration (num., iy, grad. norm, func.) :  11   0 4.777741493e-12 6.256190493e-12
                 Iteration (num., iy, grad. norm, func.) :  12   0 8.218089052e-13 6.255720318e-12
              Solving for output 0 - done. Time (sec):  0.2321289
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 9.729143239e-08 4.567639526e-08
                 Iteration (num., iy, grad. norm, func.) :   0   1 9.337808969e-08 4.538209916e-08
                 Iteration (num., iy, grad. norm, func.) :   1   1 2.904323831e-06 3.250090785e-08
                 Iteration (num., iy, grad. norm, func.) :   2   1 8.666318505e-07 4.690442846e-09
                 Iteration (num., iy, grad. norm, func.) :   3   1 2.631077034e-07 1.907895208e-09
                 Iteration (num., iy, grad. norm, func.) :   4   1 2.396633275e-07 1.714777771e-09
                 Iteration (num., iy, grad. norm, func.) :   5   1 7.155568916e-08 5.427480540e-10
                 Iteration (num., iy, grad. norm, func.) :   6   1 4.410601247e-08 4.618989947e-10
                 Iteration (num., iy, grad. norm, func.) :   7   1 1.321312805e-08 4.212892044e-10
                 Iteration (num., iy, grad. norm, func.) :   8   1 4.159525680e-09 3.621523966e-10
                 Iteration (num., iy, grad. norm, func.) :   9   1 1.402444629e-09 3.033096767e-10
                 Iteration (num., iy, grad. norm, func.) :  10   1 4.159734096e-10 2.740841148e-10
                 Iteration (num., iy, grad. norm, func.) :  11   1 3.192171188e-10 2.719205528e-10
                 Iteration (num., iy, grad. norm, func.) :  12   1 2.638366090e-10 2.718125940e-10
                 Iteration (num., iy, grad. norm, func.) :  13   1 2.313891231e-10 2.718046144e-10
                 Iteration (num., iy, grad. norm, func.) :  14   1 2.174112189e-10 2.717533849e-10
                 Iteration (num., iy, grad. norm, func.) :  15   1 6.534835588e-11 2.715115338e-10
                 Iteration (num., iy, grad. norm, func.) :  16   1 1.973765034e-11 2.713892807e-10
                 Iteration (num., iy, grad. norm, func.) :  17   1 1.795030101e-11 2.713791902e-10
                 Iteration (num., iy, grad. norm, func.) :  18   1 1.863357634e-11 2.713688461e-10
                 Iteration (num., iy, grad. norm, func.) :  19   1 1.941582151e-11 2.713636800e-10
                 Iteration (num., iy, grad. norm, func.) :  20   1 1.242443841e-11 2.713502351e-10
                 Iteration (num., iy, grad. norm, func.) :  21   1 3.885126854e-12 2.713467274e-10
                 Iteration (num., iy, grad. norm, func.) :  22   1 4.329824502e-12 2.713467255e-10
                 Iteration (num., iy, grad. norm, func.) :  23   1 3.899916065e-12 2.713462311e-10
                 Iteration (num., iy, grad. norm, func.) :  24   1 2.315857110e-12 2.713453474e-10
                 Iteration (num., iy, grad. norm, func.) :  25   1 1.562810923e-12 2.713450831e-10
                 Iteration (num., iy, grad. norm, func.) :  26   1 1.286548732e-12 2.713450522e-10
                 Iteration (num., iy, grad. norm, func.) :  27   1 1.124148706e-12 2.713450011e-10
                 Iteration (num., iy, grad. norm, func.) :  28   1 1.865113508e-12 2.713449709e-10
                 Iteration (num., iy, grad. norm, func.) :  29   1 6.271910693e-13 2.713449696e-10
              Solving for output 1 - done. Time (sec):  0.5591829
           Solving nonlinear problem (n=400) - done. Time (sec):  0.7914271
        Solving for degrees of freedom - done. Time (sec):  0.8427939
     Training - done. Time (sec):  0.8575890
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006599
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007648
     
     Prediction time/pt. (sec) :  0.0000015
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007010
     
     Prediction time/pt. (sec) :  0.0000014
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005889
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006621
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005431
     
     Prediction time/pt. (sec) :  0.0000011
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006342
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006118
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006227
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005400
     
     Prediction time/pt. (sec) :  0.0000011
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006757
     
     Prediction time/pt. (sec) :  0.0000014
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0006289
     
     Prediction time/pt. (sec) :  0.0000013
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0007188
     
     Prediction time/pt. (sec) :  0.0000014
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0005760
     
     Prediction time/pt. (sec) :  0.0000012
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0019710
     
     Prediction time/pt. (sec) :  0.0000008
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0015810
     
     Prediction time/pt. (sec) :  0.0000006
     
  
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
           Computing dof2coeff - done. Time (sec):  0.0062079
           Initializing Hessian ...
           Initializing Hessian - done. Time (sec):  0.0005131
           Computing energy terms ...
           Computing energy terms - done. Time (sec):  0.0149319
           Computing approximation terms ...
           Computing approximation terms - done. Time (sec):  0.0014241
        Pre-computing matrices - done. Time (sec):  0.0231891
        Solving for degrees of freedom ...
           Solving initial startup problem (n=1764) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.279175539e-01 1.114942861e-02
                 Iteration (num., iy, grad. norm, func.) :   0   0 1.499206775e-05 2.184788477e-08
              Solving for output 0 - done. Time (sec):  0.0275202
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 2.653045755e+00 4.799845498e+00
                 Iteration (num., iy, grad. norm, func.) :   0   1 2.441435303e-04 6.147506677e-06
              Solving for output 1 - done. Time (sec):  0.0264318
           Solving initial startup problem (n=1764) - done. Time (sec):  0.0540590
           Solving nonlinear problem (n=1764) ...
              Solving for output 0 ...
                 Iteration (num., iy, grad. norm, func.) :   0   0 8.076561564e-07 2.166293139e-08
                 Iteration (num., iy, grad. norm, func.) :   0   0 9.142058108e-07 1.723176225e-08
                 Iteration (num., iy, grad. norm, func.) :   1   0 3.513935845e-07 3.252602993e-09
                 Iteration (num., iy, grad. norm, func.) :   2   0 1.168214743e-07 1.046685700e-09
                 Iteration (num., iy, grad. norm, func.) :   3   0 6.326509912e-08 5.331049254e-10
                 Iteration (num., iy, grad. norm, func.) :   4   0 3.418239776e-08 4.111805755e-10
                 Iteration (num., iy, grad. norm, func.) :   5   0 2.256295052e-08 3.761245235e-10
                 Iteration (num., iy, grad. norm, func.) :   6   0 2.010448646e-08 3.714924187e-10
                 Iteration (num., iy, grad. norm, func.) :   7   0 1.940154325e-08 3.710813074e-10
                 Iteration (num., iy, grad. norm, func.) :   8   0 1.337930855e-08 3.606311245e-10
                 Iteration (num., iy, grad. norm, func.) :   9   0 1.491946708e-08 3.420900947e-10
                 Iteration (num., iy, grad. norm, func.) :  10   0 7.041005685e-09 3.069403043e-10
                 Iteration (num., iy, grad. norm, func.) :  11   0 2.260877881e-09 2.895211131e-10
                 Iteration (num., iy, grad. norm, func.) :  12   0 1.905111198e-09 2.891987822e-10
                 Iteration (num., iy, grad. norm, func.) :  13   0 2.373905281e-09 2.889797702e-10
                 Iteration (num., iy, grad. norm, func.) :  14   0 1.300032595e-09 2.879068517e-10
                 Iteration (num., iy, grad. norm, func.) :  15   0 1.623847751e-09 2.871275978e-10
                 Iteration (num., iy, grad. norm, func.) :  16   0 5.285589253e-10 2.868398442e-10
                 Iteration (num., iy, grad. norm, func.) :  17   0 7.922712640e-10 2.868375443e-10
                 Iteration (num., iy, grad. norm, func.) :  18   0 5.888225745e-10 2.868045063e-10
                 Iteration (num., iy, grad. norm, func.) :  19   0 1.132370943e-09 2.867143548e-10
                 Iteration (num., iy, grad. norm, func.) :  20   0 3.604109310e-10 2.866142544e-10
                 Iteration (num., iy, grad. norm, func.) :  21   0 3.554534001e-10 2.866064690e-10
                 Iteration (num., iy, grad. norm, func.) :  22   0 5.366990360e-10 2.866053737e-10
                 Iteration (num., iy, grad. norm, func.) :  23   0 6.289675890e-10 2.865891836e-10
                 Iteration (num., iy, grad. norm, func.) :  24   0 4.378955139e-10 2.865489267e-10
                 Iteration (num., iy, grad. norm, func.) :  25   0 1.379311543e-10 2.865102796e-10
                 Iteration (num., iy, grad. norm, func.) :  26   0 1.165470712e-10 2.865092134e-10
                 Iteration (num., iy, grad. norm, func.) :  27   0 1.464146843e-10 2.865071791e-10
                 Iteration (num., iy, grad. norm, func.) :  28   0 2.140809245e-10 2.865056805e-10
                 Iteration (num., iy, grad. norm, func.) :  29   0 1.779606030e-10 2.865037521e-10
                 Iteration (num., iy, grad. norm, func.) :  30   0 1.353338617e-10 2.865006425e-10
                 Iteration (num., iy, grad. norm, func.) :  31   0 1.573425416e-10 2.865006039e-10
                 Iteration (num., iy, grad. norm, func.) :  32   0 1.327422948e-10 2.864992021e-10
                 Iteration (num., iy, grad. norm, func.) :  33   0 1.054784937e-10 2.864971098e-10
                 Iteration (num., iy, grad. norm, func.) :  34   0 6.138672488e-11 2.864947948e-10
                 Iteration (num., iy, grad. norm, func.) :  35   0 6.123870556e-11 2.864945453e-10
                 Iteration (num., iy, grad. norm, func.) :  36   0 5.048804943e-11 2.864943135e-10
                 Iteration (num., iy, grad. norm, func.) :  37   0 9.399586850e-11 2.864939967e-10
                 Iteration (num., iy, grad. norm, func.) :  38   0 4.372030077e-11 2.864935099e-10
                 Iteration (num., iy, grad. norm, func.) :  39   0 5.948721280e-11 2.864932592e-10
                 Iteration (num., iy, grad. norm, func.) :  40   0 2.999872370e-11 2.864930129e-10
                 Iteration (num., iy, grad. norm, func.) :  41   0 4.787331964e-11 2.864930058e-10
                 Iteration (num., iy, grad. norm, func.) :  42   0 3.009437055e-11 2.864929464e-10
                 Iteration (num., iy, grad. norm, func.) :  43   0 3.964525862e-11 2.864928616e-10
                 Iteration (num., iy, grad. norm, func.) :  44   0 1.846213481e-11 2.864927280e-10
                 Iteration (num., iy, grad. norm, func.) :  45   0 3.500984645e-11 2.864925120e-10
                 Iteration (num., iy, grad. norm, func.) :  46   0 5.469355859e-12 2.864924656e-10
                 Iteration (num., iy, grad. norm, func.) :  47   0 5.469352970e-12 2.864924656e-10
                 Iteration (num., iy, grad. norm, func.) :  48   0 5.469343760e-12 2.864924656e-10
                 Iteration (num., iy, grad. norm, func.) :  49   0 1.251540829e-11 2.864924604e-10
                 Iteration (num., iy, grad. norm, func.) :  50   0 4.029270853e-12 2.864924394e-10
                 Iteration (num., iy, grad. norm, func.) :  51   0 1.067247644e-11 2.864924243e-10
                 Iteration (num., iy, grad. norm, func.) :  52   0 2.200754943e-12 2.864924196e-10
                 Iteration (num., iy, grad. norm, func.) :  53   0 2.199224609e-12 2.864924196e-10
                 Iteration (num., iy, grad. norm, func.) :  54   0 2.189676025e-12 2.864924196e-10
                 Iteration (num., iy, grad. norm, func.) :  55   0 3.135376413e-12 2.864924186e-10
                 Iteration (num., iy, grad. norm, func.) :  56   0 1.403223034e-12 2.864924174e-10
                 Iteration (num., iy, grad. norm, func.) :  57   0 3.378902299e-12 2.864924169e-10
                 Iteration (num., iy, grad. norm, func.) :  58   0 9.916982742e-13 2.864924163e-10
              Solving for output 0 - done. Time (sec):  2.6370471
              Solving for output 1 ...
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.342743255e-05 6.111600925e-06
                 Iteration (num., iy, grad. norm, func.) :   0   1 1.302307595e-05 5.880442538e-06
                 Iteration (num., iy, grad. norm, func.) :   1   1 1.363985006e-05 7.781714954e-07
                 Iteration (num., iy, grad. norm, func.) :   2   1 1.316050272e-05 2.746010755e-07
                 Iteration (num., iy, grad. norm, func.) :   3   1 4.268967712e-06 1.033367573e-07
                 Iteration (num., iy, grad. norm, func.) :   4   1 3.219792715e-06 6.703442433e-08
                 Iteration (num., iy, grad. norm, func.) :   5   1 2.039238895e-06 4.211149536e-08
                 Iteration (num., iy, grad. norm, func.) :   6   1 6.033889795e-07 2.515999581e-08
                 Iteration (num., iy, grad. norm, func.) :   7   1 4.726317346e-07 2.471919997e-08
                 Iteration (num., iy, grad. norm, func.) :   8   1 3.696022676e-07 2.449846324e-08
                 Iteration (num., iy, grad. norm, func.) :   9   1 2.488052165e-07 2.070463385e-08
                 Iteration (num., iy, grad. norm, func.) :  10   1 8.459610012e-08 1.660646850e-08
                 Iteration (num., iy, grad. norm, func.) :  11   1 5.658202980e-08 1.500183477e-08
                 Iteration (num., iy, grad. norm, func.) :  12   1 3.804107541e-08 1.481787787e-08
                 Iteration (num., iy, grad. norm, func.) :  13   1 3.468856986e-08 1.480601481e-08
                 Iteration (num., iy, grad. norm, func.) :  14   1 2.832164294e-08 1.477977984e-08
                 Iteration (num., iy, grad. norm, func.) :  15   1 3.791982085e-08 1.464243575e-08
                 Iteration (num., iy, grad. norm, func.) :  16   1 5.401835358e-09 1.448701556e-08
                 Iteration (num., iy, grad. norm, func.) :  17   1 6.403115896e-09 1.448630627e-08
                 Iteration (num., iy, grad. norm, func.) :  18   1 6.951789243e-09 1.448477870e-08
                 Iteration (num., iy, grad. norm, func.) :  19   1 6.270760852e-09 1.448358882e-08
                 Iteration (num., iy, grad. norm, func.) :  20   1 1.238561409e-08 1.447833451e-08
                 Iteration (num., iy, grad. norm, func.) :  21   1 2.463512479e-09 1.446909735e-08
                 Iteration (num., iy, grad. norm, func.) :  22   1 4.051882686e-09 1.446825445e-08
                 Iteration (num., iy, grad. norm, func.) :  23   1 3.421367398e-09 1.446789229e-08
                 Iteration (num., iy, grad. norm, func.) :  24   1 3.514206340e-09 1.446641231e-08
                 Iteration (num., iy, grad. norm, func.) :  25   1 1.892268440e-09 1.446502108e-08
                 Iteration (num., iy, grad. norm, func.) :  26   1 1.591286939e-09 1.446489562e-08
                 Iteration (num., iy, grad. norm, func.) :  27   1 1.843657461e-09 1.446464627e-08
                 Iteration (num., iy, grad. norm, func.) :  28   1 1.367852603e-09 1.446432383e-08
                 Iteration (num., iy, grad. norm, func.) :  29   1 1.279521953e-09 1.446413608e-08
                 Iteration (num., iy, grad. norm, func.) :  30   1 1.273827484e-09 1.446399608e-08
                 Iteration (num., iy, grad. norm, func.) :  31   1 1.069805193e-09 1.446389193e-08
                 Iteration (num., iy, grad. norm, func.) :  32   1 7.203325145e-10 1.446383478e-08
                 Iteration (num., iy, grad. norm, func.) :  33   1 1.092512544e-09 1.446377574e-08
                 Iteration (num., iy, grad. norm, func.) :  34   1 4.178765177e-10 1.446367228e-08
                 Iteration (num., iy, grad. norm, func.) :  35   1 4.695642060e-10 1.446365013e-08
                 Iteration (num., iy, grad. norm, func.) :  36   1 4.325042579e-10 1.446364130e-08
                 Iteration (num., iy, grad. norm, func.) :  37   1 5.822577989e-10 1.446363420e-08
                 Iteration (num., iy, grad. norm, func.) :  38   1 3.233691141e-10 1.446362135e-08
                 Iteration (num., iy, grad. norm, func.) :  39   1 4.820930113e-10 1.446360509e-08
                 Iteration (num., iy, grad. norm, func.) :  40   1 1.744254274e-10 1.446358617e-08
                 Iteration (num., iy, grad. norm, func.) :  41   1 2.456939839e-10 1.446357670e-08
                 Iteration (num., iy, grad. norm, func.) :  42   1 1.238702481e-10 1.446357147e-08
                 Iteration (num., iy, grad. norm, func.) :  43   1 1.543359462e-10 1.446356991e-08
                 Iteration (num., iy, grad. norm, func.) :  44   1 1.277821708e-10 1.446356861e-08
                 Iteration (num., iy, grad. norm, func.) :  45   1 1.668908600e-10 1.446356678e-08
                 Iteration (num., iy, grad. norm, func.) :  46   1 9.442875522e-11 1.446356489e-08
                 Iteration (num., iy, grad. norm, func.) :  47   1 1.282416074e-10 1.446356272e-08
                 Iteration (num., iy, grad. norm, func.) :  48   1 3.329909012e-11 1.446356055e-08
                 Iteration (num., iy, grad. norm, func.) :  49   1 3.252145554e-11 1.446356044e-08
                 Iteration (num., iy, grad. norm, func.) :  50   1 3.862212990e-11 1.446356030e-08
                 Iteration (num., iy, grad. norm, func.) :  51   1 5.495646363e-11 1.446356009e-08
                 Iteration (num., iy, grad. norm, func.) :  52   1 4.880141230e-11 1.446355966e-08
                 Iteration (num., iy, grad. norm, func.) :  53   1 1.034412694e-11 1.446355931e-08
                 Iteration (num., iy, grad. norm, func.) :  54   1 8.688965632e-12 1.446355931e-08
                 Iteration (num., iy, grad. norm, func.) :  55   1 1.416075582e-11 1.446355930e-08
                 Iteration (num., iy, grad. norm, func.) :  56   1 1.783006148e-11 1.446355927e-08
                 Iteration (num., iy, grad. norm, func.) :  57   1 1.730236391e-11 1.446355926e-08
                 Iteration (num., iy, grad. norm, func.) :  58   1 2.040787855e-11 1.446355923e-08
                 Iteration (num., iy, grad. norm, func.) :  59   1 1.406683632e-11 1.446355921e-08
                 Iteration (num., iy, grad. norm, func.) :  60   1 9.546993710e-12 1.446355919e-08
                 Iteration (num., iy, grad. norm, func.) :  61   1 8.436673636e-12 1.446355917e-08
                 Iteration (num., iy, grad. norm, func.) :  62   1 5.474457130e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  63   1 5.300214589e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  64   1 4.137292768e-12 1.446355916e-08
                 Iteration (num., iy, grad. norm, func.) :  65   1 7.384649167e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  66   1 2.714176642e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  67   1 2.900831485e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  68   1 2.763651196e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  69   1 1.639299153e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  70   1 1.613406097e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  71   1 1.850012394e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  72   1 1.313700480e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  73   1 1.860840014e-12 1.446355915e-08
                 Iteration (num., iy, grad. norm, func.) :  74   1 6.351006658e-13 1.446355915e-08
              Solving for output 1 - done. Time (sec):  2.3948679
           Solving nonlinear problem (n=1764) - done. Time (sec):  5.0320210
        Solving for degrees of freedom - done. Time (sec):  5.0861869
     Training - done. Time (sec):  5.1102870
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0018482
     
     Prediction time/pt. (sec) :  0.0000037
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013301
     
     Prediction time/pt. (sec) :  0.0000027
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009861
     
     Prediction time/pt. (sec) :  0.0000020
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0008950
     
     Prediction time/pt. (sec) :  0.0000018
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0011709
     
     Prediction time/pt. (sec) :  0.0000023
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010910
     
     Prediction time/pt. (sec) :  0.0000022
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009820
     
     Prediction time/pt. (sec) :  0.0000020
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0008998
     
     Prediction time/pt. (sec) :  0.0000018
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0010691
     
     Prediction time/pt. (sec) :  0.0000021
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0009260
     
     Prediction time/pt. (sec) :  0.0000019
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012159
     
     Prediction time/pt. (sec) :  0.0000024
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0025589
     
     Prediction time/pt. (sec) :  0.0000051
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0012889
     
     Prediction time/pt. (sec) :  0.0000026
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0013702
     
     Prediction time/pt. (sec) :  0.0000027
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0036948
     
     Prediction time/pt. (sec) :  0.0000015
     
  ___________________________________________________________________________
     
   Evaluation
     
        # eval points. : 2500
     
     Predicting ...
     Predicting - done. Time (sec):  0.0047178
     
     Prediction time/pt. (sec) :  0.0000019
     
  
.. figure:: rans_crm_wing.png
  :scale: 60 %
  :align: center
