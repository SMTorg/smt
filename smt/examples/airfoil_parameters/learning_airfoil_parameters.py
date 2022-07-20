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
