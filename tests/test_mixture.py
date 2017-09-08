"""
Test all the functions
"""
import unittest
import os

from smt.mixture import moe_main as moe
from moe_test_case import MOETestCase

import csv
import numpy as np

def construct_sample(file_name, data=True):
    """
    This function constructs x samples and y samples thanks to a csv. file
    Parameters:
    ------------
    - file_name : str
    Name of the file with samples
    - data : boolean
    Set True if the file contains y output
    Outputs:
    ------------
    - x_train : Array_like
    x samples
    - y_train : Array_like
    y samples
    """
    fichier = open(file_name)
    csvreader = csv.reader(fichier)
    x = []

    if data:
        dim = len(list(csvreader)[0]) - 1
    else:
        dim = len(list(csvreader)[0])

    fichier = open(file_name)
    csvreader = csv.reader(fichier)

    for i in range(dim):
        x.append([])
    for row in csvreader:
        for i in range(dim):
            x[i].append(float(row[i]))

    x_train = x[0]

    for i in range(1, dim):
        x_train = np.c_[x_train, x[i]]
    x_train = np.array(x_train)

    fichier.close()

    if not data:
        return x_train

    else:

        y_train = []

        fichier = open(file_name)
        csvreader = csv.reader(fichier)

        for row in csvreader:
            y_train.append(float(row[dim]))

        fichier.close()

        return x_train, y_train

PRINT_OUT = False


class TestAll(MOETestCase):
    """
    Test class
    """

    def test_all(self):
        """
        Test function
        """
        # dimension of the problem
        dim = 2

        file_name = os.path.join(os.path.dirname(
            __file__), "N1_D2_100.csv")

        x_train, y_train = construct_sample(file_name, data=True)

        # sample construction

        mixture = moe.MoE(hard_recombination=True)

        # mixture of expert
        mixture.fit(dim, x_train, y_train, y_train, detail=False,
                    plot=False, heaviside=True, number_cluster=0, median=True)

        # errors
        if mixture.hard:
            e_error = mixture.valid_hard.l_two_rel
            mse_error = mixture.valid_hard.rmse**2
        else:
            e_error = mixture.valid_smooth.l_two_rel
            mse_error = mixture.valid_smooth.rmse**2

        self.assert_error(e_error, 1e-1)

        if PRINT_OUT:
            print('%18.9e %18.9e'
                  % (e_error, mse_error))

        mixture = moe.MoE(hard_recombination=False)

        # mixture of expert
        mixture.fit(dim, x_train, y_train, y_train, detail=False,
                    plot=False, heaviside=True, number_cluster=0, median=False)

        # errors
        if mixture.hard:
            e_error = mixture.valid_hard.l_two_rel
            mse_error = mixture.valid_hard.rmse**2
        else:
            e_error = mixture.valid_smooth.l_two_rel
            mse_error = mixture.valid_smooth.rmse**2

        self.assert_error(e_error, 1e-1)

        if PRINT_OUT:
            print('%18.9e %18.9e'
                  % (e_error, mse_error))


if __name__ == '__main__':
    if PRINT_OUT:
        print('%20s %18s'
              % ('Relative_error', 'MSE_error'))
    unittest.main()
