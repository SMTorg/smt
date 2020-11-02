"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

import numpy as np
import matplotlib.gridspec as gridspec
from smt.utils.neural_net.data import random_mini_batches
from smt.utils.neural_net.optimizer import Adam
from smt.utils.neural_net.activation import Tanh, Linear
from smt.utils.neural_net.bwd_prop import L_model_backward
from smt.utils.neural_net.fwd_prop import L_model_forward, L_grads_forward
from smt.utils.neural_net.loss import lse
from smt.utils.neural_net.metrics import rsquare
from smt.utils.neural_net.data import normalize_data, load_csv


# TODO: implement batch-norm (deeper networks might suffer from exploding/vanishing gradients during training)

# ------------------------------------ S U P P O R T   F U N C T I O N S -----------------------------------------------


def initialize_parameters(layer_dims=None):
    """
    Initialize neural network given topology using "He" initialization

    :param: layer_dims: neural architecture [n_0, n_1, n_2, ..., n_L] where n = number of nodes, L = number of layer
    :param: activation: the activation function to use: tanh, sigmoid, or relu (choice dependens on problem type)
    :param: regression: True = regression problem (last layer will be linear)
                        False = classification problem (last layer will be sigmoid)

    :return: parameters: dictionary containing the neural net parameters:

                    parameters["Wl"]: matrix of weights associated with layer l
                    parameters["bl"]: vector of biases associated with layer l
                    parameters["a1"]: activation function for each layer where:    -1 -- linear activation
                                                                                    0 -- sigmoid activation
                                                                                    1 -- tanh activation
                                                                                    2 -- relu activation
    """
    if not layer_dims:
        raise Exception("Neural net does have any layers")

    # Network topology
    number_layers = len(layer_dims) - 1  # input layer doesn't count

    # Parameters
    parameters = {}
    for l in range(1, number_layers + 1):
        parameters["W" + str(l)] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) * np.sqrt(1.0 / layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


# ------------------------------------ C L A S S -----------------------------------------------------------------------


class Model(object):
    @property
    def number_of_inputs(self):
        return self._n_x

    @property
    def number_of_outputs(self):
        return self._n_y

    @property
    def number_training_examples(self):
        return self._m

    @property
    def layer_dims(self):
        return self._layer_dims

    @property
    def activations(self):
        return self._activations

    @property
    def parameters(self):
        return self._parameters

    @property
    def training_history(self):
        return self._training_history

    @property
    def scale_factors(self):
        return self._scale_factors

    @property
    def training_data(self):
        X = self._X_norm * self._scale_factors["x"][1] + self._scale_factors["x"][0]
        Y = self._Y_norm * self._scale_factors["y"][1] + self._scale_factors["y"][0]
        J = self._J_norm * self._scale_factors["y"][1] / self._scale_factors["x"][1]
        return X, Y, J

    def __init__(self, **kwargs):
        self._parameters = dict()
        self._layer_dims = list()
        self._activations = list()
        self._training_history = dict()
        self._scale_factors = {"x": (1, 1), "y": (1, 1)}
        self._X_norm = None
        self._Y_norm = None
        self._J_norm = None
        self._n_x = None
        self._n_y = None
        self._m = None
        self._caches = list()
        self._J_caches = list()

        for name, value in kwargs.items():
            setattr(self, name, value)

    @classmethod
    def initialize(cls, n_x=None, n_y=None, deep=2, wide=12):
        layer_dims = [n_x] + [wide] * deep + [n_y]
        parameters = initialize_parameters(layer_dims)
        activations = [Tanh()] * deep + [Linear()]
        attributes = {
            "_parameters": parameters,
            "_activations": activations,
            "_layer_dims": layer_dims,
            "_n_x": n_x,
            "_n_y": n_y,
        }
        return cls(**attributes)

    def load_parameters(self, parameters):
        L = len(parameters) // 2
        deep = L - 1
        wide = parameters["W1"].shape[0]
        self._n_x = parameters["W1"].shape[1]
        self._n_y = parameters["W" + str(L)].shape[0]
        self._layer_dims = [self._n_x] + [wide] * deep + [self._n_y]
        self._activations = [Tanh()] * deep + [Linear()]
        self._parameters = parameters

    def train(
        self,
        X,
        Y,
        J=None,
        num_iterations=100,
        mini_batch_size=None,
        num_epochs=1,
        alpha=0.01,
        beta1=0.9,
        beta2=0.99,
        lambd=0.0,
        gamma=0.0,
        seed=None,
        silent=False,
    ):
        """
        Train the neural network

        :param X: matrix of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples
        :param Y: matrix of shape (n_y, m) where n_y = no. of outputs
        :param J: tensor of size (n_y, n_x, m) representing the Jacobian:   dY1/dX1 = J[0][0]
                                                                            dY1/dX2 = J[0][1]
                                                                            ...
                                                                            dY2/dX1 = J[1][0]
                                                                            dY2/dX2 = J[1][1]
                                                                            ...

                   Note: to retrieve the i^th example for dY2/dX1: J[1][0][i] for all i = 1,...,m

        :param mini_batch_size: training data batches [batch_1, batch_2, ...] where batch_i = (X, Y, J)_i
        :param num_epochs: number of random passes through the entire data set (usually only used with mini-batch)
        :param alpha: learning rate
        :param beta1: parameter for ADAM optimizer
        :param beta2: parameter for ADAM optimizer
        :param lambd: regularization parameter
        :param gamma: gradient-enhancement parameter
        :param num_iterations: maximum number of optimizer iterations (per mini batch)
        :param seed: random seed in case user wants to ensure repeatability
        :param silent: don't print anything
        """
        self._load_training_data(X, Y, J)

        if not mini_batch_size:
            mini_batch_size = self.number_training_examples

        if silent:
            is_print = False
        elif mini_batch_size != 1:
            is_print = False
        else:
            is_print = True
        for e in range(num_epochs):
            self._training_history["epoch_" + str(e)] = dict()
            mini_batches = random_mini_batches(
                self._X_norm, self._Y_norm, self._J_norm, mini_batch_size, seed
            )
            for b, mini_batch in enumerate(mini_batches):

                # Get training data from this mini-batch
                X, Y, J = mini_batch

                # Optimization (learn parameters by minimizing prediction error)
                optimizer = Adam.initialize(
                    initial_guess=self._parameters,
                    cost_function=lambda p: self.cost(
                        p, self.activations, X, Y, J, lambd, gamma
                    ),
                    grad_function=lambda p: self.grad(
                        p, self.activations, X, Y, J, lambd, gamma
                    ),
                    learning_rate=alpha,
                    beta1=beta1,
                    beta2=beta2,
                )
                self._parameters = optimizer.optimize(
                    max_iter=num_iterations, is_print=is_print
                )

                # Compute average cost and print output
                avg_cost = np.mean(optimizer.cost_history).squeeze()
                self._training_history["epoch_" + str(e)][
                    "batch_" + str(b)
                ] = optimizer.cost_history

                if not silent:
                    print(
                        "epoch = {:d}, mini-batch = {:d}, avg cost = {:6.3f}".format(
                            e, b, avg_cost
                        )
                    )

    def evaluate(self, X):
        """
        Predict output(s) given inputs X.

        :param X: inputs to neural network, np array of shape (n_x, m) where n_x = no. inputs, m = no. training examples
        :return: Y: prediction, Y = np array of shape (n_y, m) where n_y = no. of outputs and m = no. of examples
        """
        assert X.shape[0] == self.number_of_inputs

        number_of_examples = X.shape[1]

        mu_x, sigma_x = self._scale_factors["x"]
        mu_y, sigma_y = self._scale_factors["y"]

        X_norm = (X - mu_x) / sigma_x

        Y_norm, _ = L_model_forward(X_norm, self.parameters, self.activations)

        Y = (Y_norm * sigma_y + mu_y).reshape(
            self.number_of_outputs, number_of_examples
        )

        return Y

    def print_parameters(self):
        """
        Print model parameters to screen for the user
        """
        for key, value in self._parameters.items():
            try:
                print("{}: {}".format(key, str(value.tolist())))
            except:
                print("{}: {}".format(key, value))

    def print_training_history(self):
        """
        Print model parameters to screen for the user
        """
        if self._training_history:
            for epoch, batches in self._training_history.items():
                for batch, history in batches.items():
                    for iteration, cost in enumerate(history):
                        print(
                            "{}, {}, iteration_{}, cost = {}".format(
                                epoch, batch, iteration, cost
                            )
                        )

    def plot_training_history(self, title="Training History", is_show_plot=True):
        """
        Plot the convergence history of the neural network learning algorithm
        """
        import matplotlib.pyplot as plt

        if self.training_history:
            if len(self.training_history.keys()) > 1:
                x_label = "epoch"
                y_label = "avg cost"
                y = []
                for epoch, batches in self.training_history.items():
                    avg_costs = []
                    for batch, values in batches.items():
                        avg_cost = np.mean(np.array(values))
                        avg_costs.append(avg_cost)
                    y.append(np.mean(np.array(avg_costs)))
                y = np.array(y)
                x = np.arange(len(y))
            elif len(self.training_history["epoch_0"]) > 1:
                x_label = "mini-batch"
                y_label = "avg cost"
                y = []
                for batch, values in self.training_history["epoch_0"].items():
                    avg_cost = np.mean(np.array(values))
                    y.append(avg_cost)
                y = np.array(y)
                x = np.arange(y.size)
            else:
                x_label = "optimizer iteration"
                y_label = "cost"
                y = np.array(self.training_history["epoch_0"]["batch_0"])
                x = np.arange(y.size)

            plt.plot(x, y)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)

            if is_show_plot:
                plt.show()

    def _load_training_data(self, X, Y, J=None):
        """
        Load and normalize training data

        :param X: matrix of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples
        :param Y: matrix of shape (n_y, m) where n_y = no. of outputs
        :param J: tensor of size (n_y, n_x, m) representing the Jacobian:   dY1/dX1 = J[0][0]
                                                                            dY1/dX2 = J[0][1]
                                                                            ...
                                                                            dY2/dX1 = J[1][0]
                                                                            dY2/dX2 = J[1][1]
                                                                            ...

                   Note: to retrieve the i^th example for dY2/dX1: J[1][0][i] for all i = 1,...,m
        """
        assert X.shape[1] == Y.shape[1]
        assert Y.shape[0] == Y.shape[0]
        assert X.shape[0] == self._n_x
        assert Y.shape[0] == self._n_y

        if J is not None:
            assert X.shape[1] == J.shape[2]
            assert X.shape[0] == J.shape[1]

        X_norm, Y_norm, J_norm, mu_x, sigma_x, mu_y, sigma_y = normalize_data(X, Y, J)
        self._X_norm = X_norm
        self._Y_norm = Y_norm
        self._J_norm = J_norm
        self._scale_factors["x"] = (mu_x, sigma_x)
        self._scale_factors["y"] = (mu_y, sigma_y)
        self._n_x, self._m = X.shape
        self._n_y = Y.shape[0]

    def cost(
        self,
        parameters,
        activations,
        x,
        y_true=None,
        dy_true=None,
        lambd=0.0,
        gamma=0.0,
    ):
        """
        Cost function for training

        :param x:
        :param parameters:
        :param activations:
        :param y_true:
        :param dy_true:
        :param lambd:
        :param gamma:
        :return:
        """
        y_pred, caches = L_model_forward(x, parameters, activations)
        dy_pred, dy_caches = L_grads_forward(x, parameters, activations)
        w = [value for name, value in parameters.items() if "W" in name]
        cost = lse(y_true, y_pred, lambd, w, dy_true, dy_pred, gamma)
        return cost

    def grad(
        self,
        parameters,
        activations,
        x,
        y_true=None,
        dy_true=None,
        lambd=0.0,
        gamma=0.0,
    ):
        """
        Gradient of cost function for training

        :param x:
        :param parameters:
        :param activations:
        :param y_true:
        :param dy_true:
        :param lambd:
        :param gamma:
        :return:
        """
        y_pred, caches = L_model_forward(x, parameters, activations)
        dy_pred, dy_caches = L_grads_forward(x, parameters, activations)
        grad = L_model_backward(
            y_pred, y_true, dy_pred, dy_true, caches, dy_caches, lambd, gamma
        )
        return grad

    def gradient(self, X):
        """
        Predict output(s) given inputs X.

        :param X: inputs to neural network, np array of shape (n_x, m) where n_x = no. inputs, m = no. training examples
        :return: J: prediction, J = np array of shape (n_y, n_x, m) = Jacobian
        """
        assert X.shape[0] == self.number_of_inputs

        number_of_examples = X.shape[1]

        mu_x, sigma_x = self._scale_factors["x"]
        mu_y, sigma_y = self._scale_factors["y"]

        X_norm = (X - mu_x) / sigma_x

        Y_norm, _ = L_model_forward(X_norm, self.parameters, self.activations)
        J_norm, _ = L_grads_forward(X_norm, self.parameters, self.activations)

        J = (J_norm * sigma_y / sigma_x).reshape(
            self.number_of_outputs, self.number_of_inputs, number_of_examples
        )

        return J

    def goodness_of_fit(self, X_test, Y_test, J_test=None, response=0, partial=0):
        import matplotlib.pyplot as plt

        assert X_test.shape[1] == Y_test.shape[1]
        assert Y_test.shape[0] == Y_test.shape[0]
        assert X_test.shape[0] == self.number_of_inputs
        assert Y_test.shape[0] == self.number_of_outputs
        if type(J_test) == np.ndarray:
            assert X_test.shape[1] == J_test.shape[2]
            assert X_test.shape[0] == J_test.shape[1]

        number_test_examples = Y_test.shape[1]

        Y_pred_test = self.evaluate(X_test)
        J_pred_test = self.gradient(X_test)

        X_train, Y_train, J_train = self.training_data

        Y_pred_train = self.evaluate(X_train)
        J_pred_train = self.gradient(X_train)

        if type(J_test) == np.ndarray:
            test = J_test[response, partial, :].reshape((1, number_test_examples))
            test_pred = J_pred_test[response, partial, :].reshape(
                (1, number_test_examples)
            )
            train = J_train[response, partial, :].reshape(
                (1, self.number_training_examples)
            )
            train_pred = J_pred_train[response, partial, :].reshape(
                (1, self.number_training_examples)
            )
            title = "Goodness of fit for dY" + str(response) + "/dX" + str(partial)
        else:
            test = Y_test[response, :].reshape((1, number_test_examples))
            test_pred = Y_pred_test[response, :].reshape((1, number_test_examples))
            train = Y_train[response, :].reshape((1, self.number_training_examples))
            train_pred = Y_pred_train[response, :].reshape(
                (1, self.number_training_examples)
            )
            title = "Goodness of fit for Y" + str(response)

        metrics = dict()
        metrics["R_squared"] = np.round(rsquare(test_pred, test), 2).squeeze()
        metrics["std_error"] = np.round(
            np.std(test_pred - test).reshape(1, 1), 2
        ).squeeze()
        metrics["avg_error"] = np.round(
            np.mean(test_pred - test).reshape(1, 1), 2
        ).squeeze()

        # Reference line
        y = np.linspace(
            min(np.min(test), np.min(train)), max(np.max(test), np.max(train)), 100
        )

        # Prepare to plot
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(title, fontsize=16)
        spec = gridspec.GridSpec(ncols=2, nrows=1, wspace=0.25)

        # Plot
        ax1 = fig.add_subplot(spec[0, 0])
        ax1.plot(y, y)
        ax1.scatter(test, test_pred, s=20, c="r")
        ax1.scatter(train, train_pred, s=100, c="k", marker="+")
        plt.legend(["perfect", "test", "train"])
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.title("RSquare = " + str(metrics["R_squared"]))

        ax2 = fig.add_subplot(spec[0, 1])
        error = (test_pred - test).T
        weights = np.ones(error.shape) / test_pred.shape[1]
        ax2.hist(error, weights=weights, facecolor="g", alpha=0.75)
        plt.xlabel("Absolute Prediction Error")
        plt.ylabel("Probability")
        plt.title(
            "$\mu$="
            + str(metrics["avg_error"])
            + ", $\sigma=$"
            + str(metrics["std_error"])
        )
        plt.grid(True)
        plt.show()

        return metrics


def run_example(
    train_csv, test_csv, inputs, outputs, partials=None
):  # pragma: no cover
    """
    Example using 2D Rastrigin function (egg-crate-looking function)

    usage:        test_model(train_csv='train_data.csv',
                            test_csv='train_data.csv',
                            inputs=["X[0]", "X[1]"],
                            outputs=["Y[0]"],
                            partials=[["J[0][0]", "J[0][1]"]])

    :param train_csv: str, csv file name containing training data
    :param test_csv: str, csv file name containing test data
    :param inputs: list(str), csv column labels corresponding to inputs
    :param outputs: list(str), csv column labels corresponding to outputs
    :param partials: list(str), csv column labels corresponding to partials
    """

    # Sample data
    X_train, Y_train, J_train = load_csv(
        file=train_csv, inputs=inputs, outputs=outputs, partials=partials
    )
    X_test, Y_test, J_test = load_csv(
        file=test_csv, inputs=inputs, outputs=outputs, partials=partials
    )

    # Hyper-parameters
    alpha = 0.05
    beta1 = 0.90
    beta2 = 0.99
    lambd = 0.1
    gamma = 1.0
    deep = 2
    wide = 12
    mini_batch_size = None  # None = use all data as one batch
    num_iterations = 25
    num_epochs = 50

    # Training
    model = Model.initialize(
        n_x=X_train.shape[0], n_y=Y_train.shape[0], deep=deep, wide=wide
    )
    model.train(
        X=X_train,
        Y=Y_train,
        J=J_train,
        alpha=alpha,
        lambd=lambd,
        gamma=gamma,
        beta1=beta1,
        beta2=beta2,
        mini_batch_size=mini_batch_size,
        num_iterations=num_iterations,
        num_epochs=num_epochs,
        silent=False,
    )
    model.plot_training_history()
    model.goodness_of_fit(
        X_test, Y_test
    )  # model.goodness_of_fit(X_test, Y_test, J_test, partial=1)


if __name__ == "__main__":  # pragma: no cover
    run_example(
        train_csv="train_data.csv",
        test_csv="train_data.csv",
        inputs=["X[0]", "X[1]"],
        outputs=["Y[0]"],
        partials=[["J[0][0]", "J[0][1]"]],
    )
