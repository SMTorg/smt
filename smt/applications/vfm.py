"""
Author: Dr. Mohamed Amine Bouhlel <mbouhlel@umich.edu>, R. Lafage

This package is distributed under New BSD license.

Variable-fidelity modeling: two types of bridge functions are available; i.e.,
additive and multiplicative
"""

import numpy as np

from smt.utils.options_dictionary import OptionsDictionary
from types import FunctionType
from smt.applications.application import SurrogateBasedApplication


class VFM(SurrogateBasedApplication):
    def _initialize(self):
        super(VFM, self)._initialize()

        declare = self.options.declare

        declare(
            "name_model_LF",
            types=object,
            values=(
                "KRG",
                "LS",
                "QP",
                "KPLS",
                "KPLSK",
                "GEKPLS",
                "RBF",
                "RMTC",
                "RMTB",
                "IDW",
            ),
            desc="Name of the low-fidelity model",
        )
        declare("options_LF", {}, types=dict, desc="Options for the low-fidelity model")
        declare(
            "name_model_bridge",
            types=object,
            values=(
                "KRG",
                "LS",
                "QP",
                "KPLS",
                "KPLSK",
                "GEKPLS",
                "RBF",
                "RMTC",
                "RMTB",
                "IDW",
            ),
            desc="Name of the bridge model",
        )
        declare("options_bridge", {}, types=dict, desc="Options for the bridge model")
        declare(
            "type_bridge",
            "Additive",
            types=str,
            values=("Additive", "Multiplicative"),
            desc="Bridge function type",
        )
        declare("X_LF", None, types=np.ndarray, desc="Low-fidelity inputs")
        declare("y_LF", None, types=np.ndarray, desc="Low-fidelity output")
        declare("X_HF", None, types=np.ndarray, desc="High-fidelity inputs")
        declare("y_HF", None, types=np.ndarray, desc="High-fidelity output")
        declare("dy_LF", None, types=np.ndarray, desc="Low-fidelity derivatives")
        declare("dy_HF", None, types=np.ndarray, desc="High-fidelity derivatives")

        self.nx = None
        self.ny = None
        self.sm_HF = None
        self._trained = False

    def predict_values(self, x):
        """
        Predict the output values at a set of points x.

        Parameters
        ----------
        x: np.ndarray[n, nx] or np.ndarray[n]
           Input values for the prediction result analysis.

        kx : int
           The 0-based index of the input variable with respect to which derivatives are desired.

        return
        ------
        y: np.ndarray
            Output values at the prediction points.

        """
        if not self._trained:
            self._apply()
        y = self.sm_HF["predict_values"](x)
        return y

    def predict_derivatives(self, x, kx):
        """
        Predict the dy_dx derivatives at a set of points.

        Parameters
        ----------
        x: np.ndarray[n, nx] or np.ndarray[n]
           Input values for the prediction result analysis.

        kx : int
           The 0-based index of the input variable with respect to which derivatives are desired.

        return
        ------
        y: np.ndarray
            Derivatives at the prediction points.

        """
        if not self._trained:
            self._apply()
        if kx is None:
            y = np.zeros(x.shape)
            for i in range(x.shape[1]):
                y[:, i] = self.sm_HF["predict_derivatives"][i](x).reshape((x.shape[0]))
        else:
            y = self.sm_HF["predict_derivatives"][kx](x).reshape((x.shape[0], self.ny))

        return y

    def _apply(self):
        """
        Algorithm of the VFM method
        """

        # For seek of readability
        if (
            self.options["X_LF"] is not None
            and self.options["y_LF"] is not None
            and self.options["X_HF"] is not None
            and self.options["y_HF"] is not None
        ):
            X_LF = self.options["X_LF"]
            y_LF = self.options["y_LF"]
            X_HF = self.options["X_HF"]
            y_HF = self.options["y_HF"]
        else:
            raise ValueError("Check X_LF, y_LF, X_HF, and y_FH")

        self.nx = X_LF.shape[1]
        self.ny = y_LF.shape[1]

        if self.options["dy_LF"] is not None:
            dy_LF = self.options["dy_LF"]
        if self.options["dy_HF"] is not None:
            dy_HF = self.options["dy_HF"]

        # Check parameters
        self._check_param()

        # Train the low fidelity model
        self.LF_deriv = self.options["options_LF"]["deriv"]
        del self.options["options_LF"]["deriv"]
        sm_LF = self.options["name_model_LF"](**self.options["options_LF"])
        sm_LF.options["print_global"] = False
        sm_LF.set_training_values(X_LF, y_LF)
        if self.LF_deriv:
            for i in range(sm_LF.nx):
                sm_LF.set_training_derivatives(X_LF, dy_LF[:, i], i)

        sm_LF.train()

        # compute the bridge data
        if self.options["type_bridge"] == "Multiplicative":
            y_bridge = y_HF / sm_LF.predict_values(X_HF)
            self.B_deriv = self.options["options_bridge"]["deriv"]
            del self.options["options_bridge"]["deriv"]
            if self.B_deriv:
                dy_bridge = np.zeros(dy_HF.shape)
                for i in range(sm_LF.nx):
                    dy_bridge[:, i] = (
                        (
                            (
                                dy_HF[:, i].reshape((y_HF.shape[0], 1))
                                * sm_LF.predict_values(X_HF)
                            )
                            - (y_HF * sm_LF.predict_derivatives(X_HF, kx=i))
                        )
                        / sm_LF.predict_values(X_HF) ** 2
                    ).reshape(y_HF.shape[0])
        elif self.options["type_bridge"] == "Additive":
            y_bridge = y_HF - sm_LF.predict_values(X_HF)
            self.B_deriv = self.options["options_bridge"]["deriv"]
            del self.options["options_bridge"]["deriv"]
            if self.B_deriv:
                dy_bridge = np.zeros(dy_HF.shape)
                for i in range(sm_LF.nx):
                    dy_bridge[:, i] = (
                        dy_HF[:, i].reshape((y_HF.shape[0], 1))
                        - sm_LF.predict_derivatives(X_HF, kx=i)
                    ).reshape((y_HF.shape[0]))
        else:
            raise ValueError("Only Additive and Multiplicative bridges are available")

        # Construct of the bridge function
        sm_bridge = self.options["name_model_bridge"](**self.options["options_bridge"])
        sm_bridge.set_training_values(X_HF, y_bridge)
        if self.B_deriv and self.options["dy_HF"] is not None:
            for i in range(sm_bridge.nx):
                sm_bridge.set_training_derivatives(X_HF, dy_bridge[:, i], i)
        sm_bridge.train()

        # Construct the final model
        sm_HF = {}
        if self.options["type_bridge"] == "Multiplicative":
            sm_HF["predict_values"] = lambda x: sm_bridge.predict_values(
                x
            ) * sm_LF.predict_values(x)
            if sm_bridge.supports["derivatives"] and sm_LF.supports["derivatives"]:
                sm_HF["predict_derivatives"] = []
                for i in range(sm_LF.nx):
                    sm_HF["predict_derivatives"].append(
                        lambda x, i=i: sm_bridge.predict_derivatives(x, i)
                        * sm_LF.predict_values(x)
                        + sm_bridge.predict_values(x) * sm_LF.predict_derivatives(x, i)
                    )

        else:
            sm_HF["predict_values"] = lambda x: sm_bridge.predict_values(
                x
            ) + sm_LF.predict_values(x)
            if sm_bridge.supports["derivatives"] and sm_LF.supports["derivatives"]:
                sm_HF["predict_derivatives"] = []
                for i in range(sm_LF.nx):
                    sm_HF["predict_derivatives"].append(
                        lambda x, i=i: sm_bridge.predict_derivatives(x, i)
                        + sm_LF.predict_derivatives(x, i)
                    )

        self._trained = True
        self.sm_HF = sm_HF

    def _check_param(self):

        """
        This function check some parameters of the model.
        """
        # Check surrogates
        if not callable(self.options["name_model_LF"]):
            if self.options["name_model_LF"] in self._surrogate_type:
                self.options["name_model_LF"] = self._surrogate_type[
                    self.options["name_model_LF"]
                ]
            else:
                raise ValueError(
                    "The LF surrogate should be one of %s, "
                    "%s was given."
                    % (self._surrogate_type.keys(), self.options["name_model_LF"])
                )

        if not callable(self.options["name_model_bridge"]):
            if self.options["name_model_bridge"] in self._surrogate_type:
                self.options["name_model_bridge"] = self._surrogate_type[
                    self.options["name_model_bridge"]
                ]
            else:
                raise ValueError(
                    "The bridge surrogate should be one of %s, "
                    "%s was given."
                    % (self._surrogate_type.keys(), self.options["name_model_bridge"])
                )

        # Initialize the parameter deriv
        if "deriv" not in self.options["options_LF"].keys():
            self.options["options_LF"]["deriv"] = False
        if "deriv" not in self.options["options_bridge"].keys():
            self.options["options_bridge"]["deriv"] = False
