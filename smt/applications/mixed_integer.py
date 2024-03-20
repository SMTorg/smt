"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.
"""

import warnings

import numpy as np

from smt.sampling_methods.sampling_method import SamplingMethod
from smt.surrogate_models.krg_based import KrgBased, MixIntKernelType
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.utils.checks import ensure_2d_array
from smt.utils.design_space import (
    BaseDesignSpace,
    CategoricalVariable,
    ensure_design_space,
)


class MixedIntegerSamplingMethod(SamplingMethod):
    """
    Sampling method decorator that takes an SMT continuous sampling method and
    cast values according x types specification to implement a sampling method
    handling integer (ORD) or categorical (ENUM) features
    """

    def __init__(self, sampling_method_class, design_space, **kwargs):
        """
        Parameters
        ----------
        sampling_method_class: class name
            SMT sampling method class
        design_space: BaseDesignSpace
            design space definition
        kwargs: options of the given sampling method
            options used to instanciate the SMT sampling method
            with the additional 'output_in_folded_space' boolean option
            specifying if doe output should be in folded space (enum indexes)
            or not (enum masks)
        """
        self._design_space = design_space
        if "random_state" in kwargs:
            self._design_space.random_state = kwargs["random_state"]
        elif self._design_space.random_state is None:
            self._design_space.random_state = 42
        self._unfolded_xlimits = design_space.get_unfolded_num_bounds()
        self._output_in_folded_space = kwargs.get("output_in_folded_space", True)
        kwargs.pop("output_in_folded_space", None)
        self._sampling_method = sampling_method_class(
            xlimits=self._unfolded_xlimits, **kwargs
        )
        super().__init__(xlimits=self._unfolded_xlimits)

    def _compute(self, nt, return_is_acting=False):
        x_doe, is_acting = self._design_space.sample_valid_x(
            nt,
            unfolded=not self._output_in_folded_space,
            random_state=self._design_space.random_state,
        )
        if return_is_acting:
            return x_doe, is_acting
        else:
            return x_doe

    def __call__(self, nt, return_is_acting=False):
        return self._compute(nt, return_is_acting)

    def expand_lhs(self, x, nt, method="basic"):
        doe = self._sampling_method(nt)
        x_doe, _ = self._design_space.correct_get_acting(doe)
        if self._output_in_folded_space:
            x_doe, _ = self._design_space.fold_x(x_doe)
        return x_doe


class MixedIntegerSurrogateModel(SurrogateModel):
    """
    Surrogate model (not Kriging) decorator that takes an SMT continuous surrogate model and
    cast values according x types specification to implement a surrogate model
    handling integer (ORD) or categorical (ENUM) features
    """

    def __init__(
        self,
        design_space,
        surrogate,
        input_in_folded_space=True,
    ):
        """
        Parameters
        ----------
        design_space: BaseDesignSpace
            design space definition
        surrogate: SMT surrogate model (not Kriging)
            instance of a SMT surrogate model
        input_in_folded_space: bool
            whether x data are in given in folded space (enum indexes) or not (enum masks)
        categorical_kernel: string
            the kernel to use for categorical inputs. Only for non continuous Kriging.
        """
        super().__init__()
        self._surrogate = surrogate
        if isinstance(self._surrogate, KrgBased):
            raise ValueError(
                "Using MixedIntegerSurrogateModel integer model with "
                + str(self._surrogate.name)
                + " is not supported. Please use MixedIntegerKrigingModel instead."
            )
        self.design_space = ensure_design_space(design_space=design_space)

        self._input_in_folded_space = input_in_folded_space
        self.supports = self._surrogate.supports
        self.options["print_global"] = False
        if "poly" in self._surrogate.options:
            if self._surrogate.options["poly"] != "constant":
                raise ValueError("constant regression must be used with mixed integer")

    @property
    def name(self):
        return "MixedInteger" + self._surrogate.name

    def _initialize(self):
        self.supports["derivatives"] = False

    def set_training_values(self, xt, yt, name=None) -> None:
        xt = ensure_2d_array(xt, "xt")

        # Round inputs
        design_space = self.design_space
        xt, _ = design_space.correct_get_acting(xt)

        if self._input_in_folded_space:
            xt_apply, _ = design_space.unfold_x(xt)
        else:
            xt_apply = xt

        super().set_training_values(xt_apply, yt, name)
        self._surrogate.set_training_values(xt_apply, yt, name)

    def update_training_values(self, yt, name=None):
        super().update_training_values(yt, name)
        self._surrogate.update_training_values(yt, name)

    def _train(self):
        self._surrogate._train()

    def predict_values(self, x: np.ndarray) -> np.ndarray:
        x_corr, is_acting = self._get_x_for_surrogate_model(x)
        return self._surrogate.predict_values(x_corr)

    def _predict_intermediate_values(self, x: np.ndarray, lvl) -> np.ndarray:
        x_corr, is_acting = self._get_x_for_surrogate_model(x)
        return self._surrogate._predict_intermediate_values(x_corr, lvl)

    def predict_variances(self, x: np.ndarray) -> np.ndarray:
        x_corr, is_acting = self._get_x_for_surrogate_model(x)
        return self._surrogate.predict_variances(x_corr)

    def predict_variances_all_levels(self, x: np.ndarray) -> np.ndarray:
        x_corr, is_acting = self._get_x_for_surrogate_model(x)
        return self._surrogate.predict_variances_all_levels(x_corr)

    def _get_x_for_surrogate_model(self, x):
        xp = ensure_2d_array(x, "xp")

        x_corr, is_acting = self.design_space.correct_get_acting(xp)
        if self._input_in_folded_space:
            x_corr, is_acting = self.design_space.unfold_x(x_corr, is_acting=is_acting)
        return x_corr, is_acting

    def _predict_values(self, x: np.ndarray) -> np.ndarray:
        pass


class MixedIntegerKrigingModel(KrgBased):
    """
    Kriging model decorator that takes an SMT continuous surrogate model and
    cast values according x types specification to implement a surrogate model
    handling integer (ORD) or categorical (ENUM) features
    """

    def __init__(
        self,
        surrogate,
        input_in_folded_space=True,
    ):
        """
        Parameters
        ----------
        surrogate: SMT Kriging surrogate model
            instance of a SMT Kriging surrogate model
        """
        super().__init__()
        self._surrogate = surrogate
        if not (isinstance(self._surrogate, KrgBased)):
            raise ValueError(
                "Using MixedIntegerKrigingModel integer model with "
                + str(self._surrogate.name)
                + " is not supported. Please use MixedIntegerSurrogateModel instead."
            )
        self.options["design_space"] = self._surrogate.design_space
        if surrogate.options["hyper_opt"] == "TNC":
            warnings.warn(
                "TNC not available yet for mixed integer handling. Switching to Cobyla"
            )

        self._surrogate.options["hyper_opt"] = "Cobyla"

        self._input_in_folded_space = input_in_folded_space
        self.supports = self._surrogate.supports
        self.options["print_global"] = False

        if "poly" in self._surrogate.options:
            if self._surrogate.options["poly"] != "constant":
                raise ValueError("constant regression must be used with mixed integer")

        design_space = self.design_space
        if (
            any(
                isinstance(dv, CategoricalVariable)
                for dv in design_space.design_variables
            )
            and self._surrogate.options["categorical_kernel"] is None
        ):
            self._surrogate.options["categorical_kernel"] = (
                MixIntKernelType.HOMO_HSPHERE
            )
            warnings.warn(
                "Using MixedIntegerSurrogateModel integer model with Continuous Relaxation is not supported. \
                    Switched to homoscedastic hypersphere kernel instead."
            )
        if self._surrogate.options["categorical_kernel"] is not None:
            self._input_in_folded_space = False

    @property
    def name(self):
        return "MixedInteger" + self._surrogate.name

    def _initialize(self):
        super()._initialize()
        self.supports["derivatives"] = False

    def set_training_values(self, xt, yt, name=None, is_acting=None):
        xt = ensure_2d_array(xt, "xt")

        # If the is_acting matrix is not given, assume input is not corrected (rounding, imputation, etc.) yet
        design_space = self.design_space
        if is_acting is None:
            xt, is_acting = design_space.correct_get_acting(xt)

        if self._input_in_folded_space:
            xt_apply, is_acting_apply = design_space.unfold_x(xt, is_acting)
        else:
            xt_apply, is_acting_apply = xt, is_acting

        super().set_training_values(xt_apply, yt, name, is_acting=is_acting_apply)
        self._surrogate.set_training_values(
            xt_apply, yt, name, is_acting=is_acting_apply
        )

    def update_training_values(self, yt, name=None):
        super().update_training_values(yt, name)
        self._surrogate.update_training_values(yt, name)

    def _train(self):
        self._surrogate._train()

    def predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        x_corr, is_acting = self._get_x_for_surrogate_model(np.copy(x))
        return self._surrogate.predict_values(x_corr, is_acting=is_acting)

    def _predict_intermediate_values(
        self, x: np.ndarray, lvl, is_acting=None
    ) -> np.ndarray:
        x_corr, is_acting = self._get_x_for_surrogate_model(x)
        return self._surrogate._predict_intermediate_values(
            x_corr, lvl, is_acting=is_acting
        )

    def predict_variances(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        x_corr, is_acting = self._get_x_for_surrogate_model(np.copy(x))
        return self._surrogate.predict_variances(x_corr, is_acting=is_acting)

    def predict_variances_all_levels(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        x_corr, is_acting = self._get_x_for_surrogate_model(x)
        return self._surrogate.predict_variances_all_levels(x_corr, is_acting=is_acting)

    def _get_x_for_surrogate_model(self, x):
        xp = ensure_2d_array(x, "xp")

        x_corr, is_acting = self.design_space.correct_get_acting(xp)
        if self._input_in_folded_space:
            x_corr, is_acting = self.design_space.unfold_x(x_corr, is_acting=is_acting)
        return x_corr, is_acting

    def _predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        pass


class MixedIntegerContext(object):
    """
    Class which acts as sampling method and surrogate model factory
    to handle integer and categorical variables consistently.
    """

    def __init__(self, design_space, work_in_folded_space=True):
        """
        Parameters
        ----------
        design_space: BaseDesignSpace
            the design space definition (includes mixed-discrete and/or hierarchical specifications)
        work_in_folded_space: bool
            whether x data are in given in folded space (enum indexes) or not (enum masks)
        """

        self._design_space = ensure_design_space(design_space=design_space)
        self._unfold_space = not work_in_folded_space
        self._unfolded_xlimits = self._design_space.get_unfolded_num_bounds()
        self._work_in_folded_space = work_in_folded_space

    @property
    def design_space(self) -> BaseDesignSpace:
        return self._design_space

    def build_sampling_method(self, random_state=None):
        """
        Build Mixed Integer LHS ESE sampler.
        """
        return_folded = self._work_in_folded_space

        def sample(n):
            x, _ = self._design_space.sample_valid_x(
                n, unfolded=not return_folded, random_state=random_state
            )
            return x

        return sample

    def build_kriging_model(self, surrogate):
        """
        Build MixedIntegerKrigingModel from given SMT surrogate model.
        """
        surrogate.options["design_space"] = self._design_space

        if surrogate.options["hyper_opt"] == "TNC":
            warnings.warn(
                "TNC not available yet for mixed integer handling. Switching to Cobyla"
            )

        surrogate.options["hyper_opt"] = "Cobyla"
        return MixedIntegerKrigingModel(
            surrogate=surrogate,
            input_in_folded_space=self._work_in_folded_space,
        )

    def build_surrogate_model(self, surrogate):
        """
        Build MixedIntegerKrigingModel from given SMT surrogate model.
        """
        return MixedIntegerSurrogateModel(
            self._design_space,
            surrogate=surrogate,
            input_in_folded_space=self._work_in_folded_space,
        )

    def get_unfolded_xlimits(self):
        """
        Returns relaxed xlimits
        Each level of an enumerate gives a new continuous dimension in [0, 1].
        Each integer dimensions are relaxed continuously.
        """

        return self._unfolded_xlimits

    def get_unfolded_dimension(self):
        """
        Returns x dimension (int) taking into account  unfolded categorical features
        """

        return len(self._unfolded_xlimits)
