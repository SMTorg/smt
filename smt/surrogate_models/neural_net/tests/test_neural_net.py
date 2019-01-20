'''
Author: Steven Berguin <<steven.berguin@gtri.gatech.edu>>

This package is distributed under New BSD license.
'''

import unittest
import smt.surrogate_models.neural_net as nn


class Test(unittest.TestCase):

    def test_activation(self):
        print("testing neural_net.activation")
        nn.activation.test_activation()

    def test_optimizer(self):
        print("testing neural_net.optimizer")
        nn.optimizer.test_optimizer()

    def test_model(self):
        print("testing neural_net.model")
        nn.model.test_model(train_csv='train_data.csv',
                            test_csv='train_data.csv',
                            inputs=["X[0]", "X[1]"],
                            outputs=["Y[0]"],
                            partials=[["J[0][0]", "J[0][1]"]])

    def test_demo(self):
        print("testing neural_net.demo")
        nn.demo.run_demo(alpha=0.1,
                         beta1=0.9,
                         beta2=0.99,
                         lambd=0.1,
                         gamma=1.0,
                         deep=3,
                         wide=6,
                         batches=32,
                         iterations=30,
                         epochs=50)

    # TODO: make test output more meaningful (e.g. print mini-batches --> mini-batch i, example j: X = ..., Y = ...)
    def test_data(self):
        print('testing neural_net.data')
        csv = 'train_data.csv'
        x_labels = ["X[0]", "X[1]"]
        y_labels = ["Y[0]"]
        dy_labels = [["J[0][0]", "J[0][1]"]]
        X, Y, J = nn.data.load_csv(file=csv, inputs=x_labels, outputs=y_labels, partials=dy_labels)
        X_norm, Y_norm, J_norm, mu_x, sigma_x, mu_y, sigma_y = nn.data.normalize_data(X, Y, J)
        mini_batches = nn.data.random_mini_batches(X_norm, Y_norm, J_norm, mini_batch_size=32, seed=1)

    # TODO: make test output more meaningful (e.g. print("expected LSE = X.XXX, computed X.XXX, error = X.XXX")
    def test_loss(self):
        print('testing neural_net.loss')
        nn.loss.test_loss()

    # TODO: add tests for fwd_prop, bwd_prop, and metrics 


if __name__ == '__main__':
    unittest.main()