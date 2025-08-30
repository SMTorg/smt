"""
Authors: Paul Saves
"""

import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from smt.surrogate_models import QP
from smt.surrogate_models.surrogate_model import SurrogateModel

from sklearn.utils import Tags


class ScikitLearnAdapter(RegressorMixin, BaseEstimator):
    """
    Adapter to wrap SMT SurrogateModel instances with scikit-learn compatible API.

    All keyword arguments passed to __init__ (besides model_cls) are treated
    as hyperparameters for the surrogate model.
    """

    def get_tags(self):
        return Tags(
            requires_y=True,
            input_tags={"sparse": False},  # indicates sparse support
            X_types=["2darray"],
        )

    def __init__(self, model_cls=QP, **model_kwargs):
        # model_cls must be a subclass of SurrogateModel
        try:
            if not issubclass(model_cls, SurrogateModel):
                raise ValueError(
                    f"model_cls must be a subclass of SurrogateModel, got {model_cls}"
                )
            # store parameters as attributes for get_params/set_params
        except TypeError:
            model_cls = QP

        self.model_cls = model_cls
        for name, value in model_kwargs.items():
            setattr(self.model_cls, name, value)
            setattr(self, name, value)

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        if issparse(X):
            X = X.toarray()
        # build surrogate model with current hyperparameters
        params_mod = self.get_params(deep=True)
        model_cls = params_mod.pop("model_cls")
        params = dict(model_cls.__dict__)
        # assemble kwargs for the surrogate
        surrogate_kwargs = {k: params[k] for k in params}
        surrogate2_kwargs = self.__dict__
        common_dict = {
            k: surrogate_kwargs[k]
            for k in surrogate_kwargs.keys() & surrogate2_kwargs.keys()
        }

        self.model_ = model_cls(**common_dict)
        X_arr = np.atleast_2d(np.asarray(X))
        self.n_features_in_ = X_arr.shape[1]
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        elif y_arr.ndim > 1:
            warnings.warn(
                "DataConversionWarning('A column-vector y"
                " was passed when a 1d array was expected",
                category=DataConversionWarning,
            )
        if len(X_arr.flatten()) < 1:
            raise ValueError(
                f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
            )

        self.model_.set_training_values(X_arr, y_arr)
        self.model_.train()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, accept_sparse=False)
        if issparse(X):
            X = X.toarray()
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            raise ValueError(
                f"Expected 2D array, got 1D array instead with shape {X_arr.shape}. "
                "Reshape your data using X.reshape(1, -1)"
            )
        X_arr = np.atleast_2d(np.asarray(X))
        self.n_features_in_ = X_arr.shape[1]

        if len(X_arr.flatten()) < 1:
            raise ValueError(
                f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
            )
        try:
            y_pred = self.model_.predict_values(X_arr)
        except AttributeError:
            raise NotFittedError("not fitted")
        # flatten if single-output
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred
