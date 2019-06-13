"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <steven.berguin@gtri.gatech.edu>

This package is distributed under New BSD license.
"""

import numpy as np
import os
import math


def load_csv(file=None, inputs=None, outputs=None, partials=None):
    """
    Load neural net training data from CSV file using numpy

    :param: file: csv filename containing training data (with headers as first row)
    :param: inputs: labels of the inputs, e.g. ["X[0]", "X[1]", "X[2]"]
    :param: outputs: labels of the inputs, e.g. ["Y[0]", "Y[1]", "Y[2]"]
    :param: partials: labels of the partials, e.g. [ ["J[0][0]", "J[0][1]", "J[0][2]"],
                                                     ["J[1][0]", "J[1][1]", "J[1][2]"],
                                                     ["J[2][0]", "J[2][1]", "J[2][2]"] ]

                Note 1: the name convention doesn't matter, but the order of the list does. Specifically,
                        the elements of the Jacobian should be listed in the same order as the elements of
                        the matrix reading from left to right, top to bottom (as shown above)
                Note 2: if the user does not provide partials (partials=None), then the model will switch to just
                        a regular, fully connected neural net without gradient-enhancement.

    :return: (X, Y, J): (np.ndarray, np.ndarray, np.ndarray) where
        X -- matrix of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples
        Y -- matrix of shape (n_y, m) where n_y = no. of outputs
        J -- tensor of size (n_y, n_x, m) representing the Jacobian: dY1/dX1 = J[0][0]
                                                                     dY1/dX2 = J[0][1]
                                                                     ...
                                                                     dY2/dX1 = J[1][0]
                                                                     dY2/dX2 = J[1][1]
                                                                     ...

                Note 3: to retrieve the i^th example for dY2/dX1: J[1][0][i] for all i = 1,...,m

    """
    if not file:
        raise Exception("No file specified")
    else:
        exists = os.path.isfile(file)
        if exists:
            headers = np.genfromtxt(file, delimiter=",", max_rows=1, dtype=str).tolist()
            data = np.genfromtxt(file, delimiter=",", skip_header=1)
            index = lambda header: headers.index(header)
        else:
            raise Exception("The file " + file + " does not exist")

    n_x = len(inputs)  # number of inputs
    n_y = len(outputs)  # number of outputs

    # Check that there are inputs and outputs
    if n_x == 0:
        raise Exception("No inputs specified")
    if n_y == 0:
        raise Exception("No outputs specified")

    m = data[:, index(inputs[0])].size  # number of examples

    X = np.zeros((n_x, m))
    for i, x_label in enumerate(inputs):
        X[i, :] = data[:, index(x_label)]

    Y = np.zeros((n_y, m))
    for i, y_label in enumerate(outputs):
        Y[i, :] = data[:, index(y_label)]

    if partials:
        J = np.zeros((n_y, n_x, m))
        if partials:
            for i, response in enumerate(partials):
                for j, dy_label in enumerate(response):
                    J[i][j] = data[:, index(dy_label)]
    else:
        J = None

    return X, Y, J


def random_mini_batches(X, Y, J, mini_batch_size=64, seed=None):
    """
    Creates a list of random minibatches from (X, Y)

    :param: X: np ndarray of size (n_x, m) containing input features of the training data
    :param: Y: np ndarray of size (n_y, m) containing output values of the training data
    :param: J: np ndarray of size (n_y, n_x, m) where m = number of examples
                                                    n_y = number of outputs
                                                    n_x = number of inputs
    :param: mini_batch_size: size of the mini-batches, integer
    :return: mini_batches: list of synchronous (mini_batch_X, mini_batch_Y, mini_batch_J)
    """
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y, J)
    permutations = list(np.random.permutation(m))
    shuffled_X = X[:, permutations].reshape(X.shape)
    shuffled_Y = Y[:, permutations].reshape(Y.shape)
    if J is not None:
        shuffled_J = J[:, :, permutations].reshape(J.shape)
    else:
        shuffled_J = None

    mini_batch_size = min(mini_batch_size, m)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        if J is not None:
            mini_batch_J = shuffled_J[
                :, :, k * mini_batch_size : (k + 1) * mini_batch_size
            ]
        else:
            mini_batch_J = None
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_J)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (k + 1) * mini_batch_size :]
        mini_batch_Y = shuffled_Y[:, (k + 1) * mini_batch_size :]
        if J is not None:
            mini_batch_J = shuffled_J[:, :, (k + 1) * mini_batch_size :]
        else:
            mini_batch_J = None
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_J)
        mini_batches.append(mini_batch)

    return mini_batches


def normalize_data(X, Y, J=None, is_classification=False):
    """
    Normalize training data to help with optimization, i.e. X_norm = (X - mu_x) / sigma_x where X is as below
                                                            Y_norm = (Y - mu_y) / sigma_y where Y is as below
                                                            J_norm = J * sigma_x/sigma_y

    Concretely, normalizing training data is essential because the neural learns by minimizing a cost function.
    Normalizing the data therefore rescales the problem in a way that aides the optimizer.

    param: X: np ndarray of input features of shape (n_x, m) where n_x = no. of inputs, m = no. of training examples
    param: Y: np ndarray of output labels of shape (n_y, m) where n_y = no. of outputs
    param: J: np ndarray of size (n_y, n_x, m) representing the Jacobian of Y w.r.t. X:

        dY1/dX1 = J[0][0]
        dY1/dX2 = J[0][1]
        ...
        dY2/dX1 = J[1][0]
        dY2/dX2 = J[1][1]
        ...

        N.B. To retrieve the i^th example for dY2/dX1: J[1][0][i] for all i = 1,...,m

    :return: X_norm, Y_norm, J_norm, mu_x, sigma_x, mu_y, sigma_y: normalized data and associated scale factors used
    """
    # Initialize
    X_norm = np.zeros(X.shape)
    Y_norm = np.zeros(Y.shape)
    if J is not None:
        J_norm = np.zeros(J.shape)
    else:
        J_norm = None

    # Dimensions
    n_x, m = X.shape
    n_y, _ = Y.shape

    # Normalize inputs
    mu_x = np.zeros((n_x, 1))
    sigma_x = np.ones((n_x, 1))
    for i in range(0, n_x):
        mu_x[i] = np.mean(X[i])
        sigma_x[i] = np.std(X[i])
        X_norm[i] = (X[i] - mu_x[i]) / sigma_x[i]

    # Normalize outputs
    mu_y = np.zeros((n_y, 1))
    sigma_y = np.ones((n_y, 1))
    if is_classification:
        Y_norm = Y  # no need to normalize {0, 1} classes
    else:
        for i in range(0, n_y):
            mu_y[i] = np.mean(Y[i])
            sigma_y[i] = np.std(Y[i])
            Y_norm[i] = (Y[i] - mu_y[i]) / sigma_y[i]

    # Normalize partials
    if J is not None:
        for i in range(0, n_y):
            for j in range(0, n_x):
                J_norm[i, j] = J[i, j] * sigma_x[j] / sigma_y[i]

    return X_norm, Y_norm, J_norm, mu_x, sigma_x, mu_y, sigma_y


if __name__ == "__main__":  # pragma: no cover

    # Check that data is read in correctly
    csv = "train_data.csv"
    x_labels = ["X[0]", "X[1]"]
    y_labels = ["Y[0]"]
    dy_labels = [["J[0][0]", "J[0][1]"]]
    X, Y, J = load_csv(file=csv, inputs=x_labels, outputs=y_labels, partials=dy_labels)

    assert X[0, 6] == 0.071429
    assert X[1, 15] == -0.821429
    assert Y[0, 21] == 7.331321
    assert J[0, 0, 57] == 51.409635
    assert J[0, 1, 209] == 59.252401

    X_norm, Y_norm, J_norm, mu_x, sigma_x, mu_y, sigma_y = normalize_data(X, Y, J)

    for i in range(X_norm.shape[1]):
        for j in range(X.shape[0]):
            assert abs(np.squeeze(X_norm[j, i] * sigma_x[j] + mu_x[j]) - X[j, i]) < 1e-6

    for i in range(Y_norm.shape[1]):
        for j in range(Y.shape[0]):
            assert abs(np.squeeze(Y_norm[j, i] * sigma_y[j] + mu_y[j]) - Y[j, i]) < 1e-6

    for i in range(J_norm.shape[2]):
        for j in range(X.shape[0]):
            for k in range(Y.shape[0]):
                assert (
                    abs(
                        np.squeeze(J_norm[k, j, i] * sigma_y[k] / sigma_x[j])
                        - J[k, j, i]
                    )
                    < 1e-6
                )

    mini_batches = random_mini_batches(
        X_norm, Y_norm, J_norm, mini_batch_size=32, seed=1
    )

    for mini_batch in mini_batches:
        X_batch, Y_batch, J_batch = mini_batch
        assert len(mini_batch) == 3
        assert X_batch.shape[0] == X.shape[0]
        assert Y_batch.shape[0] == Y.shape[0]
        assert J_batch.shape[0:2] == J.shape[0:2]
        assert X_batch.shape[1] <= 32
        assert X_batch.shape[1] == Y_batch.shape[1]
        assert X_batch.shape[1] == J_batch.shape[2]
