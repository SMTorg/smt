"""
Gradient-Enhanced Neural Networks (GENN)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

from smt.surrogate_models.surrogate_model import SurrogateModel

import numpy as np
from jenn.model import NeuralNet
from typing import Union, Dict, Tuple, List


# The missing type
SMTrainingPoints = Dict[Union[int, None], Dict[int, List[np.ndarray]]]


def _smt_to_genn(
    training_points: SMTrainingPoints,
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    """Translate data structure from SMT to GENN.

    :param training_points: training data (as per SMT API)
    :return: X, array of shape (n_x, m) where n_x = number of inputs, m = number of examples
    :return: Y, array of shape (n_y, m) where n_y = number of outputs
    :return: J, array of shape (n_y, n_x, m)
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

    if len(training_points[None]) == 1:
        return X.T, Y.T, None

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


class GENN(SurrogateModel):
    """Gradient-Enhanced Neural Net."""

    def load_data(self, xt, yt, dyt_dxt=None):
        """Load all training data into surrogate model in one step.

        :param model: SurrogateModel object for which to load training data
        :param xt: smt data points at which response is evaluated
        :param yt: response at xt
        :param dyt_dxt: gradient at xt
        """
        m, n_x = (xt.size, 1) if xt.ndim <= 1 else xt.shape
        m, n_y = (yt.size, 1) if yt.ndim <= 1 else yt.shape

        # Reshape arrays
        xt = xt.reshape((m, n_x))
        yt = yt.reshape((m, n_y))

        # Load values
        self.set_training_values(xt, yt)

        # Load partials
        if dyt_dxt is not None:
            dyt_dxt = dyt_dxt.reshape((m, n_x))
            for i in range(n_x):
                self.set_training_derivatives(xt, dyt_dxt[:, i].reshape((m, 1)), i)

    @property
    def name(self) -> str:
        return "GENN"

    def _initialize(self):
        self.supports["derivatives"] = True
        self.supports["training_derivatives"] = True

        self.options.declare(
            "alpha",
            default=0.05,
            types=(int, float),
            desc="optimizer learning rate",
        )
        self.options.declare(
            "beta1",
            default=0.9,
            types=(int, float),
            desc="Adam optimizer tuning parameter",
        )
        self.options.declare(
            "beta2",
            default=0.99,
            types=(int, float),
            desc="Adam optimizer tuning parameter",
        )
        self.options.declare(
            "lambd",
            default=0.01,
            types=(int, float),
            desc="regularization coefficient",
        )
        self.options.declare(
            "gamma",
            default=1.0,
            types=(int, float),
            desc="gradient-enhancement coefficient",
        )
        self.options.declare(
            "hidden_layer_sizes",
            default=[12, 12],
            types=list,
            desc="number of nodes per hidden layer",
        )
        self.options.declare(
            "mini_batch_size",
            default=-1,
            types=int,
            desc="split data into batches of specified size",
        )
        self.options.declare(
            "num_epochs",
            default=1,
            types=int,
            desc="number of random passes through the data",
        )
        self.options.declare(
            "num_iterations",
            default=1000,
            types=int,
            desc="number of optimizer iterations per mini-batch",
        )
        self.options.declare(
            "seed",
            default=-1,
            types=int,
            desc="random seed to control repeatability",
        )
        self.options.declare(
            "is_print",
            default=False,
            types=bool,
            desc="print progress (or not)",
        )
        self.options.declare(
            "is_normalize",
            default=False,
            types=bool,
            desc="normalize training by mean and variance",
        )
        self.options.declare(
            "is_backtracking",
            default=False,
            types=bool,
            desc="refine step step during line search (fixed otherwise)",
        )

    def _final_initialize(self):
        inputs = [1]  # will be overwritten during training (dummy value)
        output = [1]  # will be overwritten during training (dummy value)
        hidden = self.options["hidden_layer_sizes"]
        layer_sizes = inputs + hidden + output
        self.model = NeuralNet(layer_sizes)

    def _train(self):
        X, Y, J = _smt_to_genn(self.training_points)
        n_x = X.shape[0]
        n_y = Y.shape[0]
        hidden = self.options["hidden_layer_sizes"]
        layer_sizes = [n_x] + hidden + [n_y]
        self.model.parameters.layer_sizes = layer_sizes
        self.model.parameters.initialize()
        kwargs = dict(
            is_normalize=self.options["is_normalize"],
            alpha=self.options["alpha"],
            lambd=self.options["lambd"],
            gamma=self.options["gamma"],
            beta1=self.options["beta1"],
            beta2=self.options["beta2"],
            epochs=self.options["num_epochs"],
            batch_size=(
                None
                if self.options["mini_batch_size"] < 0
                else self.options["mini_batch_size"]
            ),
            max_iter=self.options["num_iterations"],
            is_backtracking=self.options["is_backtracking"],
            is_verbose=self.options["is_print"],
            random_state=None if self.options["seed"] < 0 else self.options["seed"],
        )
        self.model.fit(X, Y, J, **kwargs)

    def _predict_values(self, x):
        return self.model.predict(x.T).T

    def _predict_derivatives(self, x, kx):
        return self.model.predict_partials(x.T)[:, kx, :].T
