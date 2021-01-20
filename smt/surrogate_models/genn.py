"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.neural_net.model import Model

import numpy as np


# ------------------------------------ S U P P O R T   F U N C T I O N S -----------------------------------------------


def load_smt_data(model, xt, yt, dyt_dxt=None):
    """
    Utility function to load SMT data more easily

    :param model: SurrogateModel object for which to load training data
    :param xt: smt data points at which response is evaluated
    :param yt: response at xt
    :param dyt_dxt: gradient at xt
    """
    # Dimensionality
    if len(xt.shape) == 1:
        n_x = 1  # number of variables, x
        m = xt.size
    else:
        m, n_x = xt.shape

    if len(yt.shape) == 1:
        n_y = 1  # number of responses, y
    else:
        n_y = yt.shape[1]

    # Reshape arrays
    xt = xt.reshape((m, n_x))
    yt = yt.reshape((m, n_y))

    # Load values
    model.set_training_values(xt, yt)

    # Load partials
    if dyt_dxt is not None:
        dyt_dxt = dyt_dxt.reshape((m, n_x))
        for i in range(n_x):
            model.set_training_derivatives(xt, dyt_dxt[:, i].reshape((m, 1)), i)


def smt_to_genn(training_points):
    """
    Translate from SMT data structure to GENN data structure.

    Concretely, this neural net module works with numpy arrays in the form of (X, Y, J) as defined here-under. However,
    SMT uses a different format. Hence, we need a function that takes care of the translation.

    :param: training_points -- dict, training data in the format used by surrogate_model.py (see SMT API)

    Returns:
    :return X -- a numpy matrix of input features of shape (n_x, m) where n_x = no. of inputs, m = no. of train examples
    :return Y -- a numpy matrix of output labels of shape (n_y, m) where n_y = no. of outputs
    :return J -- a numpy array of size (n_y, n_x, m) representing the Jacobian of Y w.r.t. X:

        dY1/dX1 = J[0][0][:]
        dY1/dX2 = J[0][1][:]
        ...
        dY2/dX1 = J[1][0][:]
        dY2/dX2 = J[1][1][:]
        ...

        N.B. To retrieve the i^th example for dY2/dX1: J[1][0][i] for all i = 1,...,m

    """
    # Retrieve training data from SMT training_points
    xt, yt = training_points[None][
        0
    ]  # training_points[name][0] = [np.array(xt), np.array(yt)]

    # Deduce number of dimensions and training examples
    m, n_x = xt.shape
    _, n_y = yt.shape

    # Assign training data but transpose to match neural net implementation
    X = xt
    Y = yt

    # Loop to retrieve each partial derivative from SMT training_points
    J = np.zeros((m, n_x, n_y))
    for k in range(0, n_x):
        xt, dyt_dxt = training_points[None][k + 1]

        # assert that dimensions match
        assert xt.shape[0] == m
        assert xt.shape[1] == n_x
        assert dyt_dxt.shape[0] == m
        assert dyt_dxt.shape[1] == n_y

        # Assert that derivatives provided are for the same training points
        assert xt.all() == X.all()

        # Assign training derivative but transpose to match neural net implementation
        J[:, k, :] = dyt_dxt

    return X.T, Y.T, J.T


# ------------------------------------ C L A S S -----------------------------------------------------------------------


class GENN(SurrogateModel):
    name = "GENN"

    def _initialize(self):
        """API function: set default values for user options"""
        declare = self.options.declare
        declare("alpha", 0.5, types=(int, float), desc="optimizer learning rate")
        declare(
            "beta1", 0.9, types=(int, float), desc="Adam optimizer tuning parameter"
        )
        declare(
            "beta2", 0.99, types=(int, float), desc="Adam optimizer tuning parameter"
        )
        declare("lambd", 0.1, types=(int, float), desc="regularization coefficient")
        declare(
            "gamma", 1.0, types=(int, float), desc="gradient-enhancement coefficient"
        )
        declare("deep", 2, types=int, desc="number of hidden layers")
        declare("wide", 2, types=int, desc="number of nodes per hidden layer")
        declare(
            "mini_batch_size",
            64,
            types=int,
            desc="split data into batches of specified size",
        )
        declare(
            "num_epochs", 10, types=int, desc="number of random passes through the data"
        )
        declare(
            "num_iterations",
            100,
            types=int,
            desc="number of optimizer iterations per mini-batch",
        )
        declare(
            "seed",
            None,
            types=int,
            desc="random seed to ensure repeatability of results when desired",
        )
        declare("is_print", True, types=bool, desc="print progress (or not)")

        self.supports["derivatives"] = True
        self.supports["training_derivatives"] = True

        self.model = Model()

        self._is_trained = False

    def _train(self):
        """
        API function: train the neural net
        """
        # Convert training data to format expected by neural net module
        X, Y, J = smt_to_genn(self.training_points)

        # If there are no training derivatives, turn off gradient-enhancement
        if type(J) == np.ndarray and J.size == 0:
            self.options["gamma"] = 0.0

        # Get hyperparameters from SMT API
        alpha = self.options["alpha"]
        beta1 = self.options["beta1"]
        beta2 = self.options["beta2"]
        lambd = self.options["lambd"]
        gamma = self.options["gamma"]
        deep = self.options["deep"]
        wide = self.options["wide"]
        mini_batch_size = self.options["mini_batch_size"]
        num_iterations = self.options["num_iterations"]
        num_epochs = self.options["num_epochs"]
        seed = self.options["seed"]
        is_print = self.options["is_print"]

        # number of inputs and outputs
        n_x = X.shape[0]
        n_y = Y.shape[0]

        # Train neural net
        self.model = Model.initialize(n_x, n_y, deep, wide)
        self.model.train(
            X=X,
            Y=Y,
            J=J,
            num_iterations=num_iterations,
            mini_batch_size=mini_batch_size,
            num_epochs=num_epochs,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            lambd=lambd,
            gamma=gamma,
            seed=seed,
            silent=not is_print,
        )

        self._is_trained = True

    def _predict_values(self, x):
        """
        API method: predict values using appropriate methods from the neural_network.py module

        :param  x: np.ndarray[n, nx] -- Input values for the prediction points
        :return y: np.ndarray[n, ny] -- Output values at the prediction points
        """
        return self.model.evaluate(x.T).T

    def _predict_derivatives(self, x, kx):
        """
        API method: predict partials using appropriate methods from the neural_network.py module

        :param  x: np.ndarray[n, nx] -- Input values for the prediction points
        :param kx: int -- The 0-based index of the input variable with respect to which derivatives are desired
        :return: dy_dx: np.ndarray[n, ny] -- partial derivatives
        """
        return self.model.gradient(x.T)[:, kx, :].T

    def plot_training_history(self):
        if self._is_trained:
            self.model.plot_training_history()

    def goodness_of_fit(self, xv, yv, dyv_dxv):
        """
        Compute metrics to evaluate goodness of fit and show actual by predicted plot

        :param xv: np.ndarray[n, nx], x validation points
        :param yv: np.ndarray[n, 1], y validation response
        :param dyv_dxv: np.ndarray[n, ny], dydx validation derivatives
        """
        # Store current training points
        training_points = self.training_points

        # Replace training points with test (validation) points
        load_smt_data(self, xv, yv, dyv_dxv)

        # Convert from SMT format to a more convenient format for GENN
        X, Y, J = smt_to_genn(self.training_points)

        # Generate goodness of fit plots
        self.model.goodness_of_fit(X, Y)

        # Restore training points
        self.training_points = training_points


def run_example(is_gradient_enhancement=True):  # pragma: no cover
    """Test and demonstrate GENN using a 1D example"""
    import matplotlib.pyplot as plt

    # Test function
    f = lambda x: x * np.sin(x)
    df_dx = lambda x: np.sin(x) + x * np.cos(x)

    # Domain
    lb = -np.pi
    ub = np.pi

    # Training data
    m = 4
    xt = np.linspace(lb, ub, m)
    yt = f(xt)
    dyt_dxt = df_dx(xt)

    # Validation data
    xv = lb + np.random.rand(30, 1) * (ub - lb)
    yv = f(xv)
    dyv_dxv = df_dx(xv)

    # Initialize GENN object
    genn = GENN()
    genn.options["alpha"] = 0.1
    genn.options["beta1"] = 0.9
    genn.options["beta2"] = 0.99
    genn.options["lambd"] = 0.1
    genn.options["gamma"] = int(is_gradient_enhancement)
    genn.options["deep"] = 2
    genn.options["wide"] = 6
    genn.options["mini_batch_size"] = 64
    genn.options["num_epochs"] = 20
    genn.options["num_iterations"] = 100
    genn.options["is_print"] = True

    # Load data
    load_smt_data(genn, xt, yt, dyt_dxt)

    # Train
    genn.train()
    genn.plot_training_history()
    genn.goodness_of_fit(xv, yv, dyv_dxv)

    # Plot comparison
    if genn.options["gamma"] == 1.0:
        title = "with gradient enhancement"
    else:
        title = "without gradient enhancement"
    x = np.arange(lb, ub, 0.01)
    y = f(x)
    y_pred = genn.predict_values(x)
    fig, ax = plt.subplots()
    ax.plot(x, y_pred)
    ax.plot(x, y, "k--")
    ax.plot(xv, yv, "ro")
    ax.plot(xt, yt, "k+", mew=3, ms=10)
    ax.set(xlabel="x", ylabel="y", title=title)
    ax.legend(["Predicted", "True", "Test", "Train"])
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    run_example(is_gradient_enhancement=True)
