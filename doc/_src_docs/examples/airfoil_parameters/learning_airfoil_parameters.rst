Learning Airfoil Parameters
===========================

This is a tutorial to determine the aerodynamic coefficients of a given airfoil using GENN in SMT (other models could be used as well). 
The obtained surrogate model can be used to give predictions for certain Mach numbers, angles of attack and the aerodynamic coefficients. 
These calculations can be really useful in case of an airfoil shape optimization. The input parameters uses the airfoil Camber and Thickness mode shapes.

* Inputs: Airfoil Camber and Thickness mode shapes, Mach, alpha
* Outputs (options): cd, cl, cm

In this test case, we will be predicting only the Cd coefficient. However, the other databases for the prediction of the 
other terms are available in the same repository. Bouhlels mSANN uses the information contained in the paper [1]_ to determine 
the airfoil's mode shapes. Moreover, in mSANN a deep neural network is used to predict the Cd parameter of a given parametrized
airfoil. Therefore, in this tutorial, we reproduce the paper [2]_ using the Gradient-Enhanced Neural Networks (GENN) from SMT. 

Briefly explaining how mSANN generates the mode shapes of a given airfoil:

#. Using inverse distance weighting (IDW) to interpolate the surface function of each airfoil.
#. Then applying singular value decomposition (SVD) to reduce the number of variables that define the airfoil geometry. It includes a total of 14 airfoil modes (seven for camber and seven for thickness).
#. Totally 16 input variables, two flow conditions of Mach number (0.3 to 0.6) and the angle of attack (2 degrees to 6 degrees) plus 14 shape coefficients.
#. The output airfoil aerodynamic force coefficients and their respective gradients are computed using ADflow, which solves the RANS equations with a Spalart-Allmaras turbulence model.

References
----------

.. [1] Bouhlel, M. A., He, S., & Martins, J. R. (2020). Scalable gradient–enhanced artificial neural networks for airfoil shape design in the subsonic and transonic regimes. Structural and Multidisciplinary Optimization, 61(4), 1363-1376.
.. [2] Bouhlel, M. A., He, S., & Martins, J. R. (2019). mSANN Model Benchmarks, Mendeley Data, https://doi.org/10.17632/ngpd634smf.1.
.. [3] Li, J., Bouhlel, M. A., & Martins, J. R. (2019). Data-based approach for fast airfoil analysis and optimization. AIAA Journal, 57(2), 581-596.
.. [4] Bouhlel, M. A., & Martins, J. R. (2019). Gradient-enhanced kriging for high-dimensional problems. Engineering with Computers, 35(1), 157-173.
.. [5] Du, X., He, P., & Martins, J. R. (2021). Rapid airfoil design optimization via neural networks-based parameterization and surrogate modeling. Aerospace Science and Technology, 113, 106701.
.. [6] University of Michigan, Webfoil, 2021. URL http://webfoil.engin.umich.edu/, online accessed on 16 of June 2021.

Implementation
--------------

Utilities
^^^^^^^^^

.. code-block:: python

  import os
  import numpy as np
  import csv
  
  WORKDIR = os.path.dirname(os.path.abspath(__file__))
  
  
  def load_NACA4412_modeshapes():
      return np.loadtxt(open(os.path.join(WORKDIR, "modes_NACA4412_ct.txt")))
  
  
  def load_cd_training_data():
      with open(os.path.join(WORKDIR, "cd_x_y.csv")) as file:
          reader = csv.reader(file, delimiter=";")
          values = np.array(list(reader), dtype=np.float32)
          dim_values = values.shape
          x = values[:, : dim_values[1] - 1]
          y = values[:, -1]
      with open(os.path.join(WORKDIR, "cd_dy.csv")) as file:
          reader = csv.reader(file, delimiter=";")
          dy = np.array(list(reader), dtype=np.float32)
      return x, y, dy
  
  
  def plot_predictions(airfoil_modeshapes, Ma, cd_model):
      import matplotlib
  
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
  
      # alpha is linearily distributed over the range of -1 to 7 degrees
      # while Ma is kept constant
      inputs = np.zeros(shape=(1, 15))
      inputs[0, :14] = airfoil_modeshapes
      inputs[0, -1] = Ma
      inputs = np.tile(inputs, (50, 1))
  
      alpha = np.atleast_2d([-1 + 0.16 * i for i in range(50)]).T
  
      inputs = np.concatenate((inputs, alpha), axis=1)
  
      # Predict Cd
      cd_pred = cd_model.predict_values(inputs)
  
      # Load ADflow Cd reference
      with open(os.path.join(WORKDIR, "NACA4412-ADflow-alpha-cd.csv")) as file:
          reader = csv.reader(file, delimiter=" ")
          cd_adflow = np.array(list(reader)[1:], dtype=np.float32)
  
      plt.plot(alpha, cd_pred)
      plt.plot(cd_adflow[:, 0], cd_adflow[:, 1])
      plt.grid(True)
      plt.legend(["Surrogate", "ADflow"])
      plt.title("Drag coefficient")
      plt.xlabel("Alpha")
      plt.ylabel("Cd")
      plt.show()
  

Main
^^^^

.. code-block:: python

  """
  Predicting Airfoil Aerodynamics through data by Raul Carreira Rufato and Prof. Joseph Morlier
  """
  
  import numpy as np
  
  from smt.examples.airfoil_parameters.learning_airfoil_parameters import (
      load_cd_training_data,
      load_NACA4412_modeshapes,
      plot_predictions,
  )
  from sklearn.model_selection import train_test_split
  from smt.surrogate_models.genn import GENN
  
  x, y, dy = load_cd_training_data()
  
  # splitting the dataset
  x_train, x_test, y_train, y_test, dy_train, dy_test = train_test_split(
      x, y, dy, train_size=0.8
  )
  
  # building and training the GENN
  n_x = x_train.shape[-1]
  n_y = 1 
  
  genn = GENN(
      layer_sizes=(
          n_x,  
          6, 6,         
          n_y,  
      ), 
      print_global=False,
  )
  
  # learning rate that controls optimizer step size
  genn.options["alpha"] = 0.1
  # lambd = 0. = no regularization, lambd > 0 = regularization
  genn.options["lambd"] = 0.01
  # gamma = 0. = no grad-enhancement, gamma > 0 = grad-enhancement
  genn.options["gamma"] = 1.0
  # number of optimizer iterations per mini-batch
  genn.options["num_iterations"] = 100
  # print output (or not)
  genn.options["is_print"] = True
  # normalize training data to help convergence
  genn.options["is_normalize"] = True
  
  # convenience function to read in data that is in SMT format
  genn.load_data(x_train, y_train, dy_train)
  
  genn.train()
  
  ## non-API function to plot training history (to check convergence)
  # genn.plot_training_history()
  ## non-API function to check accuracy of regression
  # genn.goodness_of_fit(x_test, y_test, dy_test)
  
  # API function to predict values at new (unseen) points
  y_pred = genn.predict_values(x_test)
  
  # Now we will use the trained model to make a prediction with a not-learned form.
  # Example Prediction for NACA4412.
  # Airfoil mode shapes should be determined according to Bouhlel, M.A., He, S., and Martins,
  # J.R.R.A., mSANN Model Benchmarks, Mendeley Data, 2019. https://doi.org/10.17632/ngpd634smf.1
  # Comparison of results with Adflow software for an alpha range from -1 to 7 degrees. Re = 3000000
  airfoil_modeshapes = load_NACA4412_modeshapes()
  Ma = 0.3
  alpha = 0
  
  # input in neural network is created out of airfoil mode shapes, Mach number and alpha
  # airfoil_modeshapes: computed mode_shapes of random airfol geometry with parameterise_airfoil
  # Ma: desired Mach number for evaluation in range [0.3,0.6]
  # alpha: scalar in range [-1, 6]
  input = np.zeros(shape=(1, 16))
  input[0, :14] = airfoil_modeshapes
  input[0, 14] = Ma
  input[0, -1] = alpha
  
  # prediction
  cd_pred = genn.predict_values(input)
  print("Drag coefficient prediction (cd): ", cd_pred[0, 0])
  
  plot_predictions(airfoil_modeshapes, Ma, genn)
  
::

  epoch = 0, batch = 0, iter = 0, cost =  2.858
  epoch = 0, batch = 0, iter = 1, cost =  2.345
  epoch = 0, batch = 0, iter = 2, cost =  2.258
  epoch = 0, batch = 0, iter = 3, cost =  2.159
  epoch = 0, batch = 0, iter = 4, cost =  2.069
  epoch = 0, batch = 0, iter = 5, cost =  1.967
  epoch = 0, batch = 0, iter = 6, cost =  1.875
  epoch = 0, batch = 0, iter = 7, cost =  1.759
  epoch = 0, batch = 0, iter = 8, cost =  1.630
  epoch = 0, batch = 0, iter = 9, cost =  1.435
  epoch = 0, batch = 0, iter = 10, cost =  1.255
  epoch = 0, batch = 0, iter = 11, cost =  1.349
  epoch = 0, batch = 0, iter = 12, cost =  1.182
  epoch = 0, batch = 0, iter = 13, cost =  1.077
  epoch = 0, batch = 0, iter = 14, cost =  0.971
  epoch = 0, batch = 0, iter = 15, cost =  0.967
  epoch = 0, batch = 0, iter = 16, cost =  0.939
  epoch = 0, batch = 0, iter = 17, cost =  0.697
  epoch = 0, batch = 0, iter = 18, cost =  0.823
  epoch = 0, batch = 0, iter = 19, cost =  0.768
  epoch = 0, batch = 0, iter = 20, cost =  0.641
  epoch = 0, batch = 0, iter = 21, cost =  0.687
  epoch = 0, batch = 0, iter = 22, cost =  0.676
  epoch = 0, batch = 0, iter = 23, cost =  0.556
  epoch = 0, batch = 0, iter = 24, cost =  0.536
  epoch = 0, batch = 0, iter = 25, cost =  0.434
  epoch = 0, batch = 0, iter = 26, cost =  0.444
  epoch = 0, batch = 0, iter = 27, cost =  0.391
  epoch = 0, batch = 0, iter = 28, cost =  0.380
  epoch = 0, batch = 0, iter = 29, cost =  0.372
  epoch = 0, batch = 0, iter = 30, cost =  0.311
  epoch = 0, batch = 0, iter = 31, cost =  0.290
  epoch = 0, batch = 0, iter = 32, cost =  0.272
  epoch = 0, batch = 0, iter = 33, cost =  0.279
  epoch = 0, batch = 0, iter = 34, cost =  0.249
  epoch = 0, batch = 0, iter = 35, cost =  0.238
  epoch = 0, batch = 0, iter = 36, cost =  0.233
  epoch = 0, batch = 0, iter = 37, cost =  0.233
  epoch = 0, batch = 0, iter = 38, cost =  0.223
  epoch = 0, batch = 0, iter = 39, cost =  0.215
  epoch = 0, batch = 0, iter = 40, cost =  0.206
  epoch = 0, batch = 0, iter = 41, cost =  0.201
  epoch = 0, batch = 0, iter = 42, cost =  0.195
  epoch = 0, batch = 0, iter = 43, cost =  0.202
  epoch = 0, batch = 0, iter = 44, cost =  0.205
  epoch = 0, batch = 0, iter = 45, cost =  0.204
  epoch = 0, batch = 0, iter = 46, cost =  0.203
  epoch = 0, batch = 0, iter = 47, cost =  0.226
  epoch = 0, batch = 0, iter = 48, cost =  0.250
  epoch = 0, batch = 0, iter = 49, cost =  0.182
  epoch = 0, batch = 0, iter = 50, cost =  0.178
  epoch = 0, batch = 0, iter = 51, cost =  0.163
  epoch = 0, batch = 0, iter = 52, cost =  0.168
  epoch = 0, batch = 0, iter = 53, cost =  0.174
  epoch = 0, batch = 0, iter = 54, cost =  0.173
  epoch = 0, batch = 0, iter = 55, cost =  0.202
  epoch = 0, batch = 0, iter = 56, cost =  0.188
  epoch = 0, batch = 0, iter = 57, cost =  0.207
  epoch = 0, batch = 0, iter = 58, cost =  0.154
  epoch = 0, batch = 0, iter = 59, cost =  0.151
  epoch = 0, batch = 0, iter = 60, cost =  0.160
  epoch = 0, batch = 0, iter = 61, cost =  0.159
  epoch = 0, batch = 0, iter = 62, cost =  0.166
  epoch = 0, batch = 0, iter = 63, cost =  0.163
  epoch = 0, batch = 0, iter = 64, cost =  0.184
  epoch = 0, batch = 0, iter = 65, cost =  0.187
  epoch = 0, batch = 0, iter = 66, cost =  0.182
  epoch = 0, batch = 0, iter = 67, cost =  0.194
  epoch = 0, batch = 0, iter = 68, cost =  0.179
  epoch = 0, batch = 0, iter = 69, cost =  0.173
  epoch = 0, batch = 0, iter = 70, cost =  0.163
  epoch = 0, batch = 0, iter = 71, cost =  0.154
  epoch = 0, batch = 0, iter = 72, cost =  0.164
  epoch = 0, batch = 0, iter = 73, cost =  0.168
  epoch = 0, batch = 0, iter = 74, cost =  0.143
  epoch = 0, batch = 0, iter = 75, cost =  0.150
  epoch = 0, batch = 0, iter = 76, cost =  0.145
  epoch = 0, batch = 0, iter = 77, cost =  0.129
  epoch = 0, batch = 0, iter = 78, cost =  0.127
  epoch = 0, batch = 0, iter = 79, cost =  0.120
  epoch = 0, batch = 0, iter = 80, cost =  0.119
  epoch = 0, batch = 0, iter = 81, cost =  0.121
  epoch = 0, batch = 0, iter = 82, cost =  0.118
  epoch = 0, batch = 0, iter = 83, cost =  0.122
  epoch = 0, batch = 0, iter = 84, cost =  0.121
  epoch = 0, batch = 0, iter = 85, cost =  0.113
  epoch = 0, batch = 0, iter = 86, cost =  0.109
  epoch = 0, batch = 0, iter = 87, cost =  0.105
  epoch = 0, batch = 0, iter = 88, cost =  0.102
  epoch = 0, batch = 0, iter = 89, cost =  0.101
  epoch = 0, batch = 0, iter = 90, cost =  0.101
  epoch = 0, batch = 0, iter = 91, cost =  0.099
  epoch = 0, batch = 0, iter = 92, cost =  0.098
  epoch = 0, batch = 0, iter = 93, cost =  0.101
  epoch = 0, batch = 0, iter = 94, cost =  0.111
  epoch = 0, batch = 0, iter = 95, cost =  0.150
  epoch = 0, batch = 0, iter = 96, cost =  0.198
  epoch = 0, batch = 0, iter = 97, cost =  0.313
  epoch = 0, batch = 0, iter = 98, cost =  0.157
  epoch = 0, batch = 0, iter = 99, cost =  0.236
  Drag coefficient prediction (cd):  0.010507216839848385
  
.. figure:: learning_airfoil_parameters.png
  :scale: 100 %
  :align: center

