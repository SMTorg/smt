"""
This file is the interface which allows smt models to be used by MoE
"""
import numpy as np


class SMTModelAdaptor(object):
    """
    The class which adapts the SMT Model to MoE
    """

    def __init__(self, smt_model):
        """
        Attributes:
        -----------
        - smt_model: sm obect
        the smt model
        """
        self.smt_model = smt_model.__class__()
        self.smt_model.options['print_global'] = False

    def train(self, x, y):
        """
        The function train adapted for smt models
        Parameters:
        -----------
        - x: array_like
        Input sample
        - y: array_like
        Output sample
        Return:
        -----------
        the smt model trained
        """
        self.smt_model.training_pts = {'exact': {}}
        self.smt_model.set_training_values(x, y)
        return self.smt_model.train()

    def predict_output(self, x):
        """
        The function train adapted for smt models
        Parameters:
        -----------
        - x: array_like
        Input sample
        Return:
        -----------
        the output predicted
        """
        if x.ndim == 1:
            x = np.array([x])
        return self.smt_model.predict_values(x)[0]
