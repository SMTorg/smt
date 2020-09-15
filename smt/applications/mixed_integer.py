"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.
"""
import numpy as np
from smt.surrogate_models.surrogate_model import SurrogateModel
from smt.sampling_methods.sampling_method import SamplingMethod

FLOAT = "float_type"
INT = "int_type"
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
            raise ValueError(
                "Bad x limits ({}) for variable type {} (index={})".format(
                    xlimits[i], xtyp, i
                )
            )

        if (
            xtyp != FLOAT
            and xtyp != INT
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


def compute_x_unfold_dimension(xtypes):
    res = 0
    for xtyp in xtypes:
        if xtyp == FLOAT or xtyp == INT:
            res += 1
        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            res += xtyp[1]
        else:
            raise ValueError(
                "Bad var type specification: "
                "should be FLOAT, INT or (ENUM, n), got {}".format(xtyp)
            )
    return res


def unfold_to_continuous_limits(xtypes, xlimits):
    """
    Expand xlimits to add continuous dimensions for enumerate x features
    Each level of an enumerate gives a new continuous dimension in [0, 1].
    Each integer dimensions are relaxed continuously.
    
    Parameters
    ---------
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
        if xtyp == FLOAT or xtyp == INT:
            xlims.append(xlimits[i])
        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            if xtyp[1] == len(xlimits[i]):
                xlims.extend(xtyp[1] * [[0, 1]])
            else:
                raise ValueError(
                    "Bad xlimits for categorical var[{}] "
                    "should have {} categories, got only {} in {}".format(
                        i, xtyp[1], len(xlimits[i]), xlimits[i]
                    )
                )
        else:
            raise ValueError(
                "Bad var type specification: "
                "should be FLOAT, INT or (ENUM, n), got {}".format(xtyp)
            )
    return np.array(xlims)


def cast_to_discrete_values(xtypes, x):
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
    ret = np.atleast_2d(x).reshape((-1, compute_x_unfold_dimension(xtypes))).copy()
    x_col = 0
    for xtyp in xtypes:
        if xtyp == FLOAT:
            x_col += 1
            continue

        elif xtyp == INT:
            print(x)
            ret[:, x_col] = np.round(ret[:, x_col])
            x_col += 1

        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            # Categorial : The biggest level is selected.
            xenum = ret[:, x_col : x_col + xtyp[1]]
            maxx = np.max(xenum, axis=1).reshape((-1, 1))
            mask = xenum < maxx
            xenum[mask] = 0
            xenum[~mask] = 1
            x_col = x_col + xtyp[1]
        else:
            raise ValueError(
                "Bad var type specification: "
                "should be FLOAT, INT or (ENUM, n), got {}".format(xtyp)
            )
    return ret


def fold_with_enum_indexes(xtypes, x):
    """
    This function reduce categorical inputs from discrete unfolded space to  
    initial x dimension space where categorical x dimensions are valued by the index
    in the corresponding enumerate list.
    For instance, if an input dimension is typed ["blue", "red", "green"] a sample/row of 
    the input x may contain the mask [..., 0, 0, 1, ...] which will be contracted in [..., 2, ...]
    meaning the "green" value.
            
    Parameters
    ---------
    x: np.ndarray [n_evals, dim]
        continuous evaluation point input variable values 
    xlimits: np.ndarray
        the bounds of the each original dimension and their labels .
    
    Returns
    -------
    np.ndarray [n_evals, dim]
        evaluation point input variable values with enumerate index for categorical variables
    """
    xfold = np.zeros((x.shape[0], len(xtypes)))
    for i, xtyp in enumerate(xtypes):
        if xtyp == FLOAT or xtyp == INT:
            xfold[:, i] = x[:, i]
        elif isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            index = np.argmax(x[:, i : i + xtyp[1]], axis=1)
            xfold[:, i] = index
        else:
            raise ValueError(
                "Bad var type specification: "
                "should be FLOAT, INT or (ENUM, n), got {}".format(xtyp)
            )
    return xfold


class MixedIntegerSamplingMethod(SamplingMethod):
    def __init__(self, xtypes, xlimits, sampling_method_class, **kwargs):
        super()
        check_xspec_consistency(xtypes, xlimits)
        self._xtypes = xtypes
        self._xlimits = unfold_to_continuous_limits(xtypes, xlimits)
        self._sampling_method = sampling_method_class(xlimits=self._xlimits, **kwargs)

    def __call__(self, nt):
        return cast_to_discrete_values(self._xtypes, self._sampling_method(nt))


class MixedIntegerSurrogate(SurrogateModel):
    def __init__(self, xtypes, xlimits, surrogate):
        super()
        check_xspec_consistency(xtypes, xlimits)
        self._surrogate = surrogate
        self._xtypes = xtypes
        self._xlimits = xlimits
        self._fold_dim = len(xtypes)
        self._unfold_dim = compute_x_unfold_dimension(xtypes)

    def _initialize(self):
        self.supports["derivatives"] = False

    def set_training_values(self, xt, yt, name=None):
        self._surrogate.set_training_values(xt, yt, name)

    def update_training_values(self, yt, name=None):
        self._surrogate.update_training_values(yt, name)

    def train(self):
        self._surrogate.train()

    def predict_values(self, x):
        return self._surrogate.predict_values(cast_to_discrete_values(self._xtypes, x))

    def predict_variances(self, x):
        return self._surrogate.predict_variances(
            cast_to_discrete_values(self._xtypes, x)
        )


class MixedIntegerContext(object):
    def __init__(self, xtypes, xlimits):
        check_xspec_consistency(xtypes, xlimits)
        self._xtypes = xtypes
        self._xlimits = xlimits

    def build_sampling_method(self, sampling_method_class, **kwargs):
        return MixedIntegerSamplingMethod(
            self._xtypes, self._xlimits, sampling_method_class, **kwargs
        )

    def build_surrogate(self, surrogate):
        return MixedIntegerSurrogate(self._xtypes, self._xlimits, surrogate)

    def unfold_to_continuous_limits(self, xlimits):
        return unfold_to_continuous_limits(self._xtypes, xlimits)

    def cast_to_discrete_values(self, x):
        return cast_to_discrete_values(self._xtypes, x)

    def fold_with_enum_indexes(self, x):
        return fold_with_enum_indexes(self._xtypes, x)

    def cast_to_enum_values(self, x_col, enum_indexes):
        return [self._xlimits[x_col][index] for index in enum_indexes]

