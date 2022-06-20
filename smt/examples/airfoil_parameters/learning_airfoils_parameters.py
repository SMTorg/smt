import numpy as np
import matplotlib.pyplot as plt
from smt.surrogate_models.genn import GENN, load_smt_data
import csv
from sklearn.model_selection import train_test_split

"""
Predicting Airfoil Aerodynamics through data by Raul Carreira Rufato and Prof. Joseph Morlier
This is a tutorial to determine the aerodynamic coefficients of a given airfoil using GENN in SMT (other models could be used as well) 
The obtained surrogate model can be used to give predictions for certain Mach numbers, angles of attack and the aerodynamic coefficients. 
These calculations can be really useful in case of an airfoil shape optimization. The input parameters uses the airfoil Camber and Thickness mode shapes.
Inputs: Airfoil Camber and Thickness mode shapes, Mach, alpha
Outputs (options): cd, cl, cm
In this test case, we will be predicting only the Cd coefficient. However, the other databases for the prediction of the 
other terms are available in the same repository. Bouhlels mSANN uses the information contained in the paper [4] to determine 
the airfoil's mode shapes. Moreover, in mSANN a deep neural network is used to predict the Cd parameter of a given parametrized
airfoil. Therefore, in this tutorial, we reproduce the paper [1] using the Gradient-Enhanced Neural Networks (GENN)  from SMT. 
Briefly explaining how mSANN generates the mode shapes of a given airfoil:
a. Using inverse distance weighting (IDW) to interpolate the surface function of each airfoil.
b. Then applying singular value decomposition (SVD) to reduce the number of variables that define the airfoil geometry. It includes a total of 
14 airfoil modes (seven for camber and seven for thickness).
c. Totally 16 input variables, two flow conditions of Mach number (0.3 to 0.6) and the angle of attack (2 degrees to 6 degrees) plus 14 shape coefficients.
d. The output airfoil aerodynamic force coefficients and their respective gradients are computed using ADflow, which solves the RANS equations with a 
Spalart-Allmaras turbulence model.
[1] Bouhlel, M. A., He, S., and Martins, J. R. R. A., mSANN Model Benchmarks, Mendeley Data, 2019. https://doi.org/10. 17632/ngpd634smf.1.
[2] Li, J., Bouhlel,M., and Martins,J.R.R.A., Data-Based Approach for Fast Airfoil Analysis. and Optimization,AIAA Journal, 2019.
[3] Bouhlel, M.A., and Martins,J.R.R.A.,Gradient-EnhancedKrigingforHigh-DimensionalProblems,Springer-Verlag,2018.
[4] Bouhlel,M., He,S., and Martins,J.R.R.A.,Scalable Gradient-Enhanced Artificial Neural Networks for Airfoil Shape Design in the Subsonic and Transonic 
Regimes, ResearchGate, 2020.
[5] Du,X.,He,P.,and Martins,J.,Rapid airfoil design optimization via neuralnetworks-basedp arameterization and surrogate modeling, Elsevier, 2021.
[6] University of Michigan, Webfoil, 2021. URL http://webfoil.engin.umich.edu/, online accessed on 16 of June 2021.
"""

# getting datasets
def getData():
    with open("dataSMTCd.csv") as file:
        reader = csv.reader(file, delimiter=";")
        values = np.array(list(reader), dtype=np.float32)
        dim_values = values.shape
        x = values[:, : dim_values[1] - 1]
        y = values[:, -1]
    with open("DataDySMTCd.csv") as file:
        reader = csv.reader(file, delimiter=";")
        dy = np.array(list(reader), dtype=np.float32)
    return x, y, dy


def graphPredictionsSMT(airfoil_modeshapes, airfoil_name, Ma, plot: bool, genn):
    # loading of the models
    modelcd = genn
    # input arrays are created -> alpha is linearily distributed over the range of -2 to 6 degrees while Ma is kept constant
    input_array = np.zeros(shape=(1, 15))
    input_array[0, :14] = airfoil_modeshapes
    input_array[0, -1] = Ma
    new_input_array = np.zeros(shape=(1, 15))
    new_input_array[0, :14] = airfoil_modeshapes
    new_input_array[0, -1] = Ma
    for i in range(0, 49):
        new_input_array = np.concatenate((new_input_array, input_array), axis=0)
    alpha = np.zeros(shape=(50, 1))
    for i in range(0, 50):
        alpha[i, 0] = -2 + 0.16 * i
    input_array = np.concatenate((new_input_array, alpha), axis=1)
    # predictions are made
    cd_pred = modelcd.predict_values(input_array)
    # graphs for the single aerodynamic coefficients are computed -> through bool: plot it is to decide if graphs are computed or not
    if plot == True:
        x, y_comp = reconstruct_airfoil(airfoil_modeshapes)
        plt.plot(x, y_comp)
        plt.axis([-0.1, 1.2, -0.6, 0.6])
        plt.grid(True)
        plt.title(airfoil_name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"Airfoil_{airfoil_name}")
        plt.close()
        plt.plot(alpha, cd_pred)
        plt.grid(True)
        plt.title("Drag coefficient SMT")
        plt.xlabel("Alpha")
        plt.ylabel("Cd")
        plt.savefig(f"SMT_Cd_{airfoil_name}")
        plt.close()
    # array for the aerodynamic coeffs and alpha and the airfoil name
    return cd_pred, alpha, airfoil_name


# to get predictions first the models must be trained and saved
# all links must be changed to your file path
# airfoil_modeshapes: computed mode_shapes of random airfol geometry with parameterise_airfoil
# Ma: desired Mach number for evaluation in range [0.3,0.6]
# gives a scalar prediction using the models trained in SMT
# alpha scalar in range [-1, 6]
def scalarPredictionsSMT(airfoil_modeshapes, Ma, alpha, genn):
    # loading of the models (not yet a direct function in SMT, that is why the way over pickle)
    modelcd = genn
    # input array in neural network is created out of airfoil mode shapes, Mach number and alpha
    input_array = np.zeros(shape=(1, 16))
    input_array[0, :14] = airfoil_modeshapes
    input_array[0, 14] = Ma
    input_array[0, -1] = alpha
    # predictions are made
    cd_pred = modelcd.predict_values(input_array)
    return cd_pred


# gives an array of predicted aerodynamic coefficients
# plot: bool if graph should be created or not -> if None; just returns the arrays for the aerodynamic coefficients
# airfoil_name: string
# alpha scalar in range [-1, 10]
def graphPredictionsSMT(airfoil_modeshapes, airfoil_name, Ma, plot: bool, genn):
    # loading of the models
    modelcd = genn
    # input arrays are created -> alpha is linearily distributed over the range of -1 to 7 degrees while Ma is kept constant
    input_array = np.zeros(shape=(1, 15))
    input_array[0, :14] = airfoil_modeshapes
    input_array[0, -1] = Ma
    new_input_array = np.zeros(shape=(1, 15))
    new_input_array[0, :14] = airfoil_modeshapes
    new_input_array[0, -1] = Ma
    for i in range(0, 49):
        new_input_array = np.concatenate((new_input_array, input_array), axis=0)
    alpha = np.zeros(shape=(50, 1))
    for i in range(0, 50):
        alpha[i, 0] = -1 + 0.16 * i
    input_array = np.concatenate((new_input_array, alpha), axis=1)
    # predictions are made
    cd_pred = modelcd.predict_values(input_array)
    with open("NACA4412-ADflow-alpha-cd.csv") as file:
        reader = csv.reader(file, delimiter=" ")
        cd_adflow = np.array(list(reader)[1:], dtype=np.float32)
    # cd from ADflow and Xfoil
    # graphs for the single aerodynamic coefficients are computed -> through bool: plot it is to decide if graphs are computed or not
    if plot == True:
        x, y_comp = reconstruct_airfoil(airfoil_modeshapes)
        plt.figure()
        plt.plot(x, y_comp)
        plt.axis([-0.1, 1.2, -0.6, 0.6])
        plt.grid(True)
        plt.title(airfoil_name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.figure()
        plt.plot(alpha, cd_pred)
        plt.plot(cd_adflow[:, 0], cd_adflow[:, 1])
        plt.grid(True)
        plt.legend(["Surrogate", "ADflow"])
        plt.title("Drag coefficient")
        plt.xlabel("Alpha")
        plt.ylabel("Cd")
    # array for the aerodynamic coeffs and alpha and the airfoil name
    return cd_pred, alpha, airfoil_name


def reconstruct_airfoil(airfoil_modes):
    modes = np.loadtxt(open("modes.txt"))
    # modes from the Database:
    # Bouhlel, M. A., He, S., and Martins, J. R. R. A., “mSANN Model Benchmarks,” Mendeley Data, 2019. https://doi.org/10.17632/ngpd634smf.1
    # the x-vector for the distribution of the points of the airfoil geometry is saved in the first line of the mode_matrix
    x = modes[0, :].copy()
    mode_matrix = modes[1:, :].copy()
    # computing the y-values of the airfoil using the mode shapes and the mode_matrix
    y_comp = np.dot(airfoil_modes, mode_matrix).flatten()
    return x, y_comp


x, y, dy = getData()
# splitting the dataset
x_train, x_test, y_train, y_test, dy_train, dy_test = train_test_split(
    x, y, dy, train_size=0.8
)
# building and training the GENN
genn = GENN()
genn.options["alpha"] = 0.001  # learning rate that controls optimizer step size
genn.options[
    "lambd"
] = 0.1  # lambd = 0. = no regularization, lambd > 0 = regularization
genn.options[
    "gamma"
] = 1.0  # gamma = 0. = no grad-enhancement, gamma > 0 = grad-enhancement
genn.options["deep"] = 2  # number of hidden layers
genn.options["wide"] = 6  # number of nodes per hidden layer
genn.options[
    "mini_batch_size"
] = 256  # used to divide data into training batches (use for large data sets)
genn.options["num_epochs"] = 25  # number of passes through data
genn.options["num_iterations"] = 10  # number of optimizer iterations per mini-batch
genn.options["is_print"] = True  # print output (or not)
load_smt_data(
    genn, x_train, y_train, dy_train
)  # convenience function to read in data that is in SMT format
genn.train()
genn.plot_training_history()  # non-API function to plot training history (to check convergence)
genn.goodness_of_fit(
    x_test, y_test, dy_test
)  # non-API function to check accuracy of regression
y_pred = genn.predict_values(
    x_test
)  # API function to predict values at new (unseen) points

# Now we will use the trained model to make a prediction with an unlearned form.
# Example Prediction for NACA4412.
# Airfoil mode shapes should be determined according to Bouhlel, M.A., He, S., and Martins,
# J.R.R.A., mSANN Model Benchmarks, Mendeley Data, 2019. https://doi.org/10.17632/ngpd634smf.1
# Comparison of results with Adflow software for an alpha range from -1 to 7 degrees. Re = 3000000
NACA4412 = np.loadtxt(open("NACA4412.txt"))
airfoil_modeshapes = np.loadtxt(open("modes_NACA4412_ct.txt"))
Ma = 0.3
alpha = 0
cd_pred = scalarPredictionsSMT(airfoil_modeshapes, Ma, alpha, genn)
print("Drag coeffitient prediction (cd): ", cd_pred[0, 0])
cd_pred, alpha, airfoil_name = graphPredictionsSMT(
    airfoil_modeshapes, "NACA4412", Ma, True, genn
)
