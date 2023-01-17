"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.
"""

from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.sampling_methods.sampling_method import SamplingMethod
from smt.utils.checks import ensure_2d_array
from smt.utils.mixed_integer import (
    cast_to_discrete_values,
    cast_to_enum_value,
    cast_to_mixed_integer,
    encode_with_enum_index,
    fold_with_enum_index,
    unfold_with_enum_mask,
    unfold_xlimits_with_continuous_limits,
)
from smt.surrogate_models.krg_based import KrgBased


class MixedIntegerSamplingMethod(SamplingMethod):
    """
    Sampling method decorator that takes an SMT continuous sampling method and
    cast values according x types specification to implement a sampling method
    handling integer (ORD) or categorical (ENUM) features
    """

    def __init__(self, sampling_method_class, xspecs=None, **kwargs):
        """
        Parameters
        ----------
        sampling_method_class: class name
            SMT sampling method class
        kwargs: options of the given sampling method
            options used to instanciate the SMT sampling method
            with the additional 'output_in_folded_space' boolean option
            specifying if doe output should be in folded space (enum indexes)
            or not (enum masks)
        """
        super()
        if hasattr(self, "_xspecs") and xspecs is None:
            xspecs = self._xspecs
        self._xspecs = xspecs
        self._xspecs.check_xspec_consistency()
        self._unfolded_xlimits = unfold_xlimits_with_continuous_limits(self._xspecs)
        self._output_in_folded_space = kwargs.get("output_in_folded_space", True)
        kwargs.pop("output_in_folded_space", None)
        self._sampling_method = sampling_method_class(
            xlimits=self._unfolded_xlimits, **kwargs
        )

    def _compute(self, nt):
        doe = self._sampling_method(nt)
        unfold_xdoe = cast_to_discrete_values(self._xspecs, True, doe)
        if self._output_in_folded_space:
            return fold_with_enum_index(self._xspecs["xtypes"], unfold_xdoe)
        else:
            return unfold_xdoe

    def __call__(self, nt):
        return self._compute(nt)


class MixedIntegerSurrogateModel(SurrogateModel):
    """
    Surrogate model (not Kriging) decorator that takes an SMT continuous surrogate model and
    cast values according x types specification to implement a surrogate model
    handling integer (ORD) or categorical (ENUM) features
    """

    def __init__(
        self,
        surrogate,
        input_in_folded_space=True,
        categorical_kernel=None,
        cat_kernel_comps=None,
        xspecs=None,
    ):
        """
        Parameters
        ----------
        xspecs : x specifications { "xlimits": xlimits, "xtypes": xtypes }
            xtypes: x types list
                x type specification: list of either FLOAT, ORD or (ENUM, n) spec.
            xlimits: array-like
                bounds of x features
        surrogate: SMT surrogate model (not Kriging)
            instance of a SMT surrogate model
        input_in_folded_space: bool
            whether x data are in given in folded space (enum indexes) or not (enum masks)
        categorical_kernel: string
            the kernel to use for categorical inputs. Only for non continuous Kriging.
        """
        super().__init__()
        self._surrogate = surrogate
        self._categorical_kernel = categorical_kernel
        self._cat_kernel_comps = cat_kernel_comps

        if isinstance(self._surrogate, KrgBased):
            raise ValueError(
                "Using MixedIntegerSurrogateModel integer model with "
                + str(self._surrogate.name)
                + " is not supported. Please use MixedIntegerKrigingModel instead."
            )
        self._xspecs = xspecs
        self._xspecs.check_xspec_consistency()

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

    def set_training_values(self, xt, yt, name=None):

        xt = ensure_2d_array(xt, "xt")
        if self._input_in_folded_space:
            xt2 = unfold_with_enum_mask(self._xspecs["xtypes"], xt)
        else:
            xt2 = xt
        xt2 = cast_to_discrete_values(
            self._xspecs, (self._categorical_kernel == None), xt2
        )
        super().set_training_values(xt2, yt)
        self._surrogate.set_training_values(xt2, yt, name)

    def update_training_values(self, yt, name=None):
        super().update_training_values(yt, name)
        self._surrogate.update_training_values(yt, name)

    def _train(self):
        self._surrogate._train()

    def predict_values(self, x):
        xp = ensure_2d_array(x, "xp")
        if self._input_in_folded_space:
            x2 = unfold_with_enum_mask(self._xspecs["xtypes"], xp)
        else:
            x2 = xp
        return self._surrogate.predict_values(
            cast_to_discrete_values(
                self._xspecs, (self._categorical_kernel == None), x2
            )
        )

    def predict_variances(self, x):
        xp = ensure_2d_array(x, "xp")
        if self._input_in_folded_space:
            x2 = unfold_with_enum_mask(self._xspecs["xtypes"], xp)
        else:
            x2 = xp
        return self._surrogate.predict_variances(
            cast_to_discrete_values(
                self._xspecs, (self._categorical_kernel == None), x2
            )
        )

    def _predict_values(self, x):
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
        categorical_kernel=None,
        cat_kernel_comps=None,
        xspecs=None,
    ):
        """
        Parameters
        ----------
        xspecs : x specifications { "xlimits": xlimits, "xtypes": xtypes }
            xtypes: x types list
                x type specification: list of either FLOAT, ORD or (ENUM, n) spec.
            xlimits: array-like
                bounds of x features
        surrogate: SMT surrogate model
            instance of a SMT surrogate model
        input_in_folded_space: bool
            whether x data are in given in folded space (enum indexes) or not (enum masks)
        categorical_kernel: string
            the kernel to use for categorical inputs. Only for non continuous Kriging.
        """
        super().__init__()
        self._surrogate = surrogate
        self._categorical_kernel = categorical_kernel
        self._cat_kernel_comps = cat_kernel_comps
        if not (isinstance(self._surrogate, KrgBased)):
            raise ValueError(
                "Using MixedIntegerKrigingModel integer model with "
                + str(self._surrogate.name)
                + " is not supported. Please use MixedIntegerSurrogateModel instead."
            )

        if xspecs is None or xspecs["xlimits"] is None or xspecs["xtypes"] is None:
            self._xspecs = self._surrogate.options["xspecs"]
        else:
            self._xspecs = xspecs
        self._xspecs.check_xspec_consistency()
        self._input_in_folded_space = input_in_folded_space
        self.supports = self._surrogate.supports
        self.options["print_global"] = False

        if "poly" in self._surrogate.options:
            if self._surrogate.options["poly"] != "constant":
                raise ValueError("constant regression must be used with mixed integer")

        if self._categorical_kernel is not None:
            if self._surrogate.name not in ["Kriging", "KPLS"]:
                raise ValueError("matrix kernel not implemented for this model")
            if self._xspecs["xtypes"] is None:
                raise ValueError("xtypes mandatory for categorical kernel")
            self._input_in_folded_space = False

        if (
            self._surrogate.name in ["Kriging", "KPLS"]
            and self._categorical_kernel is not None
        ):
            self._surrogate.options["categorical_kernel"] = self._categorical_kernel
            if self._cat_kernel_comps is not None:
                self._surrogate.options["cat_kernel_comps"] = self._cat_kernel_comps
            self._xspecs["xtypes"] = self._surrogate.options["xspecs"]["xtypes"]

    @property
    def name(self):
        return "MixedInteger" + self._surrogate.name

    def _initialize(self):
        self.supports["derivatives"] = False

    def set_training_values(self, xt, yt, name=None):

        xt = ensure_2d_array(xt, "xt")
        if self._input_in_folded_space:
            xt2 = unfold_with_enum_mask(self._xspecs["xtypes"], xt)
        else:
            xt2 = xt
        xt2 = cast_to_discrete_values(
            self._xspecs, (self._categorical_kernel == None), xt2
        )
        super().set_training_values(xt2, yt)
        self._surrogate.set_training_values(xt2, yt, name)

    def update_training_values(self, yt, name=None):
        super().update_training_values(yt, name)
        self._surrogate.update_training_values(yt, name)

    def _train(self):
        self._surrogate._train()

    def predict_values(self, x):
        xp = ensure_2d_array(x, "xp")
        if self._input_in_folded_space:
            x2 = unfold_with_enum_mask(self._xspecs["xtypes"], xp)
        else:
            x2 = xp
        return self._surrogate.predict_values(
            cast_to_discrete_values(
                self._xspecs, (self._categorical_kernel == None), x2
            )
        )

    def predict_variances(self, x):
        xp = ensure_2d_array(x, "xp")
        if self._input_in_folded_space:
            x2 = unfold_with_enum_mask(self._xspecs["xtypes"], xp)
        else:
            x2 = xp
        return self._surrogate.predict_variances(
            cast_to_discrete_values(
                self._xspecs, (self._categorical_kernel == None), x2
            )
        )

    def _predict_values(self, x):
        pass


class MixedIntegerContext(object):
    """
    Class which acts as sampling method and surrogate model factory
    to handle integer and categorical variables consistently.
    """

    def __init__(
        self,
        xspecs={"xtypes": None, "xlimits": None},
        work_in_folded_space=True,
        categorical_kernel=None,
        cat_kernel_comps=None,
    ):
        """
        Parameters
        ----------
        xspecs : x specifications (xtypes,xlimits)
            xtypes: x types list
                x types specification: list of either FLOAT, ORD or (ENUM, n) spec.
            xlimits: array-like
                bounds of x features
        work_in_folded_space: bool
            whether x data are in given in folded space (enum indexes) or not (enum masks)
        categorical_kernel: string
            the kernel to use for categorical inputs. Only for non continuous Kriging.
        """
        self._categorical_kernel = categorical_kernel
        self._cat_kernel_comps = cat_kernel_comps
        self._xspecs = xspecs
        if xspecs["xlimits"] is not None and xspecs["xtypes"] is not None:
            self._xspecs.check_xspec_consistency()
            self._unfolded_xlimits = unfold_xlimits_with_continuous_limits(
                self._xspecs, unfold_space=(self._categorical_kernel == None)
            )
        self._work_in_folded_space = work_in_folded_space

    def build_sampling_method(self, sampling_method_class, xspecs=None, **kwargs):
        """
        Build MixedIntegerSamplingMethod from given SMT sampling method.
        """
        if xspecs is None:
            xspecs = self._xspecs
        else:
            self._xspecs = xspecs
        kwargs["output_in_folded_space"] = self._work_in_folded_space
        return MixedIntegerSamplingMethod(sampling_method_class, xspecs, **kwargs)

    def build_kriging_model(self, surrogate):
        """
        Build MixedIntegerKrigingModel from given SMT surrogate model.
        """
        return MixedIntegerKrigingModel(
            surrogate=surrogate,
            input_in_folded_space=self._work_in_folded_space,
            categorical_kernel=self._categorical_kernel,
            cat_kernel_comps=self._cat_kernel_comps,
            xspecs=self._xspecs,
        )

    def build_surrogate_model(self, surrogate):
        """
        Build MixedIntegerKrigingModel from given SMT surrogate model.
        """
        return MixedIntegerSurrogateModel(
            surrogate=surrogate,
            input_in_folded_space=self._work_in_folded_space,
            categorical_kernel=self._categorical_kernel,
            cat_kernel_comps=self._cat_kernel_comps,
            xspecs=self._xspecs,
        )

    def get_unfolded_xlimits(self):
        """
        Returns relaxed xlimits
        Each level of an enumerate gives a new continuous dimension in [0, 1].
        Each integer dimensions are relaxed continuously.
        """
        if not (hasattr(self, "_unfolded_xlimits")):
            self._unfolded_xlimits = unfold_xlimits_with_continuous_limits(
                self._xspecs, unfold_space=(self._categorical_kernel == None)
            )
        return self._unfolded_xlimits

    def get_unfolded_dimension(self):
        """
        Returns x dimension (int) taking into account  unfolded categorical features
        """
        if not (hasattr(self, "_unfolded_xlimits")):
            self._unfolded_xlimits = unfold_xlimits_with_continuous_limits(
                self._xspecs, unfold_space=(self._categorical_kernel == None)
            )
        return len(self._unfolded_xlimits)

    def cast_to_discrete_values(self, x):
        """
        Project continuously relaxed values to their closer assessable values.
        Note: categorical (or enum) x dimensions are still expanded that is
        there are still as many columns as categorical possible values for the given x dimension.
        For instance, if an input dimension is typed ["blue", "red", "green"] in xlimits a sample/row of
        the input x may contain the values (or mask) [..., 0, 0, 1, ...] to specify "green" for
        this original dimension.

        Parameters
        ----------
        x : np.ndarray [n_evals, dim]
            continuous evaluation point input variable values

        Returns
        -------
        np.ndarray
            feasible evaluation point value in categorical space.
        """
        return cast_to_discrete_values(
            self._xspecs, (self._categorical_kernel == None), x
        )

    def fold_with_enum_index(self, x):
        """
        Reduce categorical inputs from discrete unfolded space to
        initial x dimension space where categorical x dimensions are valued by the index
        in the corresponding enumerate list.
        For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
        the input x may contain the mask [..., 0, 0, 1, ...] which will be contracted in [..., 2, ...]
        meaning the "green" value.
        This function is the opposite of unfold_with_enum_mask().

        Parameters
        ----------
        x: np.ndarray [n_evals, dim]
            continuous evaluation point input variable values

        Returns
        -------
        np.ndarray [n_evals, dim]
            evaluation point input variable values with enumerate index for categorical variables
        """
        return fold_with_enum_index(self._xspecs["xtypes"], x)

    def unfold_with_enum_mask(self, x):
        """
        Expand categorical inputs from initial x dimension space where categorical x dimensions
        are valued by the index in the corresponding enumerate list to the discrete unfolded space.
        For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of
        the input x may contain [..., 2, ...] which will be expanded in [..., 0, 0, 1, ...].
        This function is the opposite of fold_with_enum_index().

        Parameters
        ----------
        x: np.ndarray [n_evals, nx]
            continuous evaluation point input variable values

        Returns
        -------
        np.ndarray [n_evals, nx continuous]
            evaluation point input variable values with enumerate index for categorical variables
        """
        return unfold_with_enum_mask(self._xspecs["xtypes"], x)

    def cast_to_enum_value(self, x_col, enum_indexes):
        """
        Return enumerate levels from indexes for the given x feature specified by x_col.

        Parameters
        ----------
        x_col: int
            index of the feature typed as enum
        enum_indexes: list
            list of indexes in the possible values for the enum

        Returns
        -------
            list of levels (labels) for the given enum feature
        """
        return cast_to_enum_value(self._xspecs, x_col, enum_indexes)

    def cast_to_mixed_integer(self, x):
        """
        Convert an x point with enum indexes to x point with enum levels

        Parameters
        ----------
        x: array-like
            point to convert

        Returns
        -------
            x as a list with enum levels if any
        """
        return cast_to_mixed_integer(self._xspecs, x)

    def encode_with_enum_index(self, x):
        """
        Convert an x point with enum levels to x point with enum indexes

        Parameters
        ----------
        x as a list with enum levels if any
            point to convert
        Returns
        -------
        np.ndarray [n_evals, dim]
            evaluation point input variable values with enumerate index for categorical variables
        """

        return encode_with_enum_index(self._specs, x)
