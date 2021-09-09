"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.
"""

import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.sampling_methods.sampling_method import SamplingMethod
from smt.utils.checks import ensure_2d_array
from smt.utils.misc import take_closest_in_list
from smt.surrogate_models.krg_based import GOWER

FLOAT = "float_type"
INT = "int_type"
ORD = "ord_type"
ENUM = "enum_type"


def check_xspec_consistency(xtypes, xlimits):
    if len(xlimits) != len(xtypes):
        raise ValueError(
            "number of x limits ({}) do not"
            "correspond to number of specified types ({})".format(
                len(xlimits), len(xtypes)
            )
        )

    for i, xtyp in enumerate(xtypes):
        if (not isinstance(xtyp, tuple)) and len(xlimits[i]) != 2:
            if xtyp == ORD and isinstance(xlimits[i][0], str):
                listint = list(map(float, xlimits[i]))
                sortedlistint = sorted(listint)
                if not np.array_equal(sortedlistint, listint):
                    raise ValueError(
                        "Unsorted x limits ({}) for variable type {} (index={})".format(
                            xlimits[i], xtyp, i
                        )
                    )

            else:
                raise ValueError(
                    "Bad x limits ({}) for variable type {} (index={})".format(
                        xlimits[i], xtyp, i
                    )
                )
        if xtyp == INT:
            if not isinstance(xlimits[i][0], str):
                xtyp = ORD
                xtypes[i] = ORD
            else:
                raise ValueError(
                    "INT do not work with list of ordered values, use ORD instead"
                )
        if (
            xtyp != FLOAT
            and xtyp != ORD
            and (not isinstance(xtyp, tuple) or xtyp[0] != ENUM)
        ):
            raise ValueError("Bad type specification {}".format(xtyp))

        if isinstance(xtyp, tuple) and len(xlimits[i]) != xtyp[1]:
            raise ValueError(
                "Bad x limits and x types specs not consistent. "
                "Got a categorical type with {} levels "
                "while x limits contains {} values (index={})".format(
                    xtyp[1], len(xlimits[i]), i
                )
            )


def _raise_value_error(xtyp):
    raise ValueError(
        "Bad xtype specification: "
        "should be FLOAT, ORD or (ENUM, n), got {}".format(xtyp)
    )


def compute_unfolded_dimension(xtypes):
    """
    Returns x dimension (int) taking into account  unfolded categorical features
    """
    res = 0
    for xtyp in xtypes:
        if xtyp == FLOAT or xtyp == ORD:
            res += 1
        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            res += xtyp[1]
        else:
            _raise_value_error(xtyp)
    return res


def unfold_xlimits_with_continuous_limits(xtypes, xlimits, categorical_kernel=None):
    """
    Expand xlimits to add continuous dimensions for enumerate x features
    Each level of an enumerate gives a new continuous dimension in [0, 1].
    Each integer dimensions are relaxed continuously.

    Parameters
    ----------
    xtypes: x types list
        x type specification
    xlimits : list
        bounds of each original dimension (bounds of enumerates being the list of levels).

    Returns
    -------
    np.ndarray [nx continuous, 2]
        bounds of the each dimension where limits for enumerates (ENUM)
        are expanded ([0, 1] for each level).
    """
    # Continuous optimization : do nothing
    xlims = []
    for i, xtyp in enumerate(xtypes):
        if xtyp == FLOAT or xtyp == ORD:
            k = xlimits[i][0]
            if xtyp == ORD and (not isinstance(xlimits[i][0], int)):
                listint = list(map(float, xlimits[i]))
                listint = [listint[0], listint[-1]]
                xlims.append(listint)
            else:
                xlims.append(xlimits[i])
        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            if xtyp[1] == len(xlimits[i]):
                if categorical_kernel is None:
                    xlims.extend(xtyp[1] * [[0, 1]])
                else:
                    listint = list(map(float, [0, len(xlimits[i])]))
                    listint = [listint[0], listint[-1]]
                    xlims.append(listint)
            else:
                raise ValueError(
                    "Bad xlimits for categorical var[{}] "
                    "should have {} categories, got only {} in {}".format(
                        i, xtyp[1], len(xlimits[i]), xlimits[i]
                    )
                )
        else:
            _raise_value_error(xtyp)
    return np.array(xlims).astype(float)


def cast_to_discrete_values(xtypes, xlimits, categorical_kernel, x):
    """
    see MixedIntegerContext.cast_to_discrete_values
    """
    ret = ensure_2d_array(x, "x").copy()
    x_col = 0
    for i, xtyp in enumerate(xtypes):
        if xtyp == FLOAT:
            x_col += 1
            continue
        elif xtyp == ORD:
            if isinstance(xlimits[i][0], str):
                listint = list(map(float, xlimits[i]))
                ret[:, x_col] = take_closest_in_list(listint, ret[:, x_col])
            else:
                ret[:, x_col] = np.round(ret[:, x_col])
            x_col += 1
        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            if categorical_kernel is None:
                # Categorial : The biggest level is selected.
                xenum = ret[:, x_col : x_col + xtyp[1]]
                maxx = np.max(xenum, axis=1).reshape((-1, 1))
                mask = xenum < maxx
                xenum[mask] = 0
                xenum[~mask] = 1
                x_col = x_col + xtyp[1]
            else:
                ret[:, x_col] = np.round(ret[:, x_col])
                x_col += 1
        else:
            _raise_value_error(xtyp)
    return ret


def fold_with_enum_index(xtypes, x, categorical_kernel=None):
    """
    see MixedIntegerContext.fold_with_enum_index
    """
    x = np.atleast_2d(x)
    xfold = np.zeros((x.shape[0], len(xtypes)))
    unfold_index = 0
    for i, xtyp in enumerate(xtypes):
        if xtyp == FLOAT or xtyp == ORD:
            xfold[:, i] = x[:, unfold_index]
            unfold_index += 1
        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            index = np.argmax(x[:, unfold_index : unfold_index + xtyp[1]], axis=1)
            xfold[:, i] = index
            unfold_index += xtyp[1]
        else:
            _raise_value_error(xtyp)
    return xfold


def unfold_with_enum_mask(xtypes, x):
    """
    see MixedIntegerContext.unfold_with_enum_mask
    """
    x = np.atleast_2d(x)
    xunfold = np.zeros((x.shape[0], compute_unfolded_dimension(xtypes)))
    unfold_index = 0
    for i, xtyp in enumerate(xtypes):
        if xtyp == FLOAT or xtyp == ORD:
            xunfold[:, unfold_index] = x[:, i]
            unfold_index += 1
        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            enum_slice = xunfold[:, unfold_index : unfold_index + xtyp[1]]
            for row in range(x.shape[0]):
                enum_slice[row, x[row, i].astype(int)] = 1
            unfold_index += xtyp[1]
        else:
            _raise_value_error(xtyp)
    return xunfold


def cast_to_enum_value(xlimits, x_col, enum_indexes):
    """
    see MixedIntegerContext.cast_to_enum_value
    """
    return [xlimits[x_col][index] for index in enum_indexes]


def cast_to_mixed_integer(xtypes, xlimits, x):
    """
    see MixedIntegerContext.cast_to_mixed_integer
    """
    res = []
    for i, xtyp in enumerate(xtypes):
        xi = x[i]
        if xtyp == FLOAT:
            res.append(xi)
        elif xtyp == ORD:
            res.append(int(xi))
        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            res.append(xlimits[i][int(xi)])
        else:
            _raise_value_error(xtyp)
    return res


class MixedIntegerSamplingMethod(SamplingMethod):
    """
    Sampling method decorator that takes an SMT continuous sampling method and
    cast values according x types specification to implement a sampling method
    handling integer (ORD) or categorical (ENUM) features
    """

    def __init__(self, xtypes, xlimits, sampling_method_class, **kwargs):
        """
        Parameters
        ----------
        xtypes: x types list
            x types specification
        xlimits: array-like
            bounds of x features
        sampling_method_class: class name
            SMT sampling method class
        kwargs: options of the given sampling method
            options used to instanciate the SMT sampling method
            with the additional 'output_in_folded_space' boolean option
            specifying if doe output should be in folded space (enum indexes)
            or not (enum masks)
        """
        super()
        check_xspec_consistency(xtypes, xlimits)
        self._xtypes = xtypes
        self._xlimits = xlimits
        self._unfolded_xlimits = unfold_xlimits_with_continuous_limits(
            self._xtypes, xlimits
        )
        self._output_in_folded_space = kwargs.get("output_in_folded_space", True)
        kwargs.pop("output_in_folded_space", None)
        self._sampling_method = sampling_method_class(
            xlimits=self._unfolded_xlimits, **kwargs
        )

    def _compute(self, nt):
        doe = self._sampling_method(nt)
        unfold_xdoe = cast_to_discrete_values(self._xtypes, self._xlimits, None, doe)
        if self._output_in_folded_space:
            return fold_with_enum_index(self._xtypes, unfold_xdoe)
        else:
            return unfold_xdoe

    def __call__(self, nt):
        return self._compute(nt)


class MixedIntegerSurrogateModel(SurrogateModel):
    """
    Surrogate model decorator that takes an SMT continuous surrogate model and
    cast values according x types specification to implement a surrogate model
    handling integer (ORD) or categorical (ENUM) features
    """

    def __init__(
        self,
        xtypes,
        xlimits,
        surrogate,
        input_in_folded_space=True,
        categorical_kernel=None,
    ):
        """
        Parameters
        ----------
        xtypes: x types list
            x type specification
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
        check_xspec_consistency(xtypes, xlimits)
        self._surrogate = surrogate
        self._categorical_kernel = categorical_kernel
        self._xtypes = xtypes
        self._xlimits = xlimits
        self._input_in_folded_space = input_in_folded_space
        self.supports = self._surrogate.supports
        self.options["print_global"] = False

        if "poly" in self._surrogate.options:
            if self._surrogate.options["poly"] != "constant":
                raise ValueError("constant regression must be used with mixed integer")

        if not (self._categorical_kernel is None):
            if self._surrogate.name not in ["Kriging"]:
                raise ValueError("matrix kernel not implemented for this model")
            if self._surrogate.options["corr"] in ["matern32", "matern52"]:
                raise ValueError("matrix kernel not compatible with matern kernel")
            if self._xtypes is None:
                raise ValueError("xtypes mandatory for categorical kernel")
            bool_raise = False
            for xtype in self._xtypes:
                if xtype == FLOAT or xtype == ORD:
                    bool_raise = True
                else:
                    if bool_raise:
                        raise ValueError(
                            "please write ENUM before FLOAT/ORD with categorical kernel"
                        )

            self._input_in_folded_space = False
        if self._surrogate.name in ["Kriging"] and self._categorical_kernel is not None:
            self._surrogate.options["categorical_kernel"] = self._categorical_kernel
            self._surrogate.options["xtypes"] = self._xtypes

    @property
    def name(self):
        return "MixedInteger" + self._surrogate.name

    def _initialize(self):
        self.supports["derivatives"] = False

    def set_training_values(self, xt, yt, name=None):

        xt = ensure_2d_array(xt, "xt")
        if self._input_in_folded_space:
            xt2 = unfold_with_enum_mask(self._xtypes, xt)
        else:
            xt2 = xt
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
            x2 = unfold_with_enum_mask(self._xtypes, xp)
        else:
            x2 = xp
        return self._surrogate.predict_values(
            cast_to_discrete_values(
                self._xtypes, self._xlimits, self._categorical_kernel, x2
            )
        )

    def predict_variances(self, x):
        xp = ensure_2d_array(x, "xp")
        if self._input_in_folded_space:
            x2 = unfold_with_enum_mask(self._xtypes, xp)
        else:
            x2 = xp
        return self._surrogate.predict_variances(
            cast_to_discrete_values(
                self._xtypes, self._xlimits, self._categorical_kernel, x2
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
        xtypes,
        xlimits,
        work_in_folded_space=True,
        categorical_kernel=None,
    ):
        """
        Parameters
        ----------
        xtypes: x types list
            x type specification: list of either FLOAT, ORD or (ENUM, n) spec.
        xlimits: array-like
            bounds of x features
        work_in_folded_space: bool
            whether x data are in given in folded space (enum indexes) or not (enum masks)
        categorical_kernel: string
            the kernel to use for categorical inputs. Only for non continuous Kriging.
        """
        check_xspec_consistency(xtypes, xlimits)
        self._xtypes = xtypes
        self._xlimits = xlimits
        self._categorical_kernel = categorical_kernel
        self._unfolded_xlimits = unfold_xlimits_with_continuous_limits(
            self._xtypes, xlimits, categorical_kernel
        )
        self._work_in_folded_space = work_in_folded_space

    def build_sampling_method(self, sampling_method_class, **kwargs):
        """
        Build MixedIntegerSamplingMethod from given SMT sampling method.
        """
        kwargs["output_in_folded_space"] = self._work_in_folded_space
        return MixedIntegerSamplingMethod(
            self._xtypes, self._xlimits, sampling_method_class, **kwargs
        )

    def build_surrogate_model(self, surrogate):
        """
        Build MixedIntegerSurrogateModel from given SMT surrogate model.
        """
        return MixedIntegerSurrogateModel(
            xtypes=self._xtypes,
            xlimits=self._xlimits,
            surrogate=surrogate,
            input_in_folded_space=self._work_in_folded_space,
            categorical_kernel=self._categorical_kernel,
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
            self._xtypes, self._xlimits, self._categorical_kernel, x
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
        return fold_with_enum_index(self._xtypes, x)

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
        return unfold_with_enum_mask(self._xtypes, x)

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
        return cast_to_enum_value(self._xlimits, x_col, enum_indexes)

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
        return cast_to_mixed_integer(self._xtypes, self._xlimits, x)
