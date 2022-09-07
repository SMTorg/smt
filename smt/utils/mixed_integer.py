"""
Author: Remi Lafage <remi.lafage@onera.fr>

This package is distributed under New BSD license.
"""

import numpy as np

from smt.utils.checks import ensure_2d_array
from smt.utils.misc import take_closest_in_list

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
                if isinstance(x[row, i], np.ndarray):
                    enum_slice[row, x[row, i].astype(int)] = 1
                else:
                    enum_slice[row, int(x[row, i])] = 1
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


def encode_with_enum_index(xtypes, xlimits, x):
    """
    see MixedIntegerContext.encode_with_enum_index
    """
    res = []
    for i, xtyp in enumerate(xtypes):
        xi = x[i]
        if isinstance(xtyp, tuple) and xtyp[0] == ENUM:
            res.append(xlimits[i].index(xi))
        elif xtyp == ORD or xtyp == FLOAT:
            res.append(xi)
        else:
            _raise_value_error(xtyp)
    return np.array(res)
