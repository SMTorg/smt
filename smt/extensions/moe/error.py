"""
This files calculates, stocks and prints the different type of errors
"""
import math
import numpy as np
from numpy import fabs
from numpy import linalg as LA


class Error(object):
    """
    This Class contains errors :
    - l_two : float
    L2 error
    - l_two_rel : float
    relative L2 error
    - mse : float
    mse error
    - rmse : float
    rmse error
    - lof : float
    lof error
    - r_two : float
    Residual
    - err_rel: array_like
    relative errors table
    - err_rel_mean : float
    mean of err_rel
    -err_rel_max : float
    max of err_rel
    -err_abs_max: flot
    max of err_abs (norm inf)
    """

    def __init__(self, y_array_true, y_array_calc):
        length = len(y_array_true)
        self.l_two = np.linalg.norm((y_array_true - y_array_calc), 2)
        self.l_two_rel = self.l_two / np.linalg.norm((y_array_true), 2)
        self.mse = (self.l_two**2) / length
        self.rmse = math.sqrt(self.mse)
        err = fabs((y_array_true - y_array_calc)) / \
            fabs(y_array_true)
        self.err_rel = 100 * err
        self.err_rel_mean = np.mean(self.err_rel)
        self.err_rel_max = max(self.err_rel)
        self.err_abs_max = LA.norm((y_array_true - y_array_calc), np.inf)
        self.quant = QuantError(err)
        if abs(np.var(y_array_true)) > 1e-10:
            self.lof = 100 * self.mse / np.var(y_array_true)
            self.r_two = (1 - self.lof / 100)
        else:
            self.lof = None
            self.r_two = None

    def show(self):  # pragma: no cover
        """
        Method to print the different types of errors
        """
        print "ERROR:"
        print "#######"
        print "l_two:", self.l_two
        print "l_two_rel", self.l_two_rel
        print "rmse:", self.rmse
        print "lof:", self.lof
        print "r_two:", self.r_two
        print "err_rel_mean:", self.err_rel_mean
        print "err_rel_max:", self.err_rel_max
        print "err_abs_max:", self.err_abs_max
        self.quant.show()


class QuantError (object):
    """
    This Class contains :
    - quant : Quant_error
    Quantiles of errors
    """

    def __init__(self, x):
        x_abs = fabs(x)
        x_abs_sort = np.sort(x_abs)
        lenght = len(x)
        ind50 = math.ceil(0.5 * lenght) - 1
        ind80 = math.ceil(0.8 * lenght) - 1
        ind90 = math.ceil(0.9 * lenght) - 1
        ind95 = math.ceil(0.95 * lenght) - 1
        ind99 = math.ceil(0.99 * lenght) - 1
        ind999 = math.ceil(0.999 * lenght) - 1
        self.val_50 = x_abs_sort[int(ind50)]
        self.val_80 = x_abs_sort[int(ind80)]
        self.val_90 = x_abs_sort[int(ind90)]
        self.val_95 = x_abs_sort[int(ind95)]
        self.val_99 = x_abs_sort[int(ind99)]
        self.val_999 = x_abs_sort[int(ind999)]
        self.pro_50 = 100 * sum(x_abs_sort <= 0.5) / lenght
        self.pro_80 = 100 * sum(x_abs_sort <= 0.2) / lenght
        self.pro_90 = 100 * sum(x_abs_sort <= 0.1) / lenght
        self.pro_95 = 100 * sum(x_abs_sort <= 0.05) / lenght
        self.pro_99 = 100 * sum(x_abs_sort <= 0.01) / lenght
        self.pro_999 = 100 * sum(x_abs_sort <= 0.001) / lenght

    def show(self):  # pragma: no cover
        """
        Method to print the different quantiles of errors
        """
        print "QUANT:"
        print "#######"
        print "fiftypercent:", self.val_50
        print "twentypercent:", self.val_80
        print "tenpercent:", self.val_90
        print "fivepercent:", self.val_95
        print "onepercent:", self.val_99
        print "onepointpercent:", self.val_999
        print "#######"
        print "QUANT_ABS:"
        print "#######"
        print "q999:", self.pro_999
        print "q99:", self.pro_99
        print "q95", self.pro_95
        print "q90", self.pro_90
        print "q80:", self.pro_80
        print "q50:", self.pro_50
