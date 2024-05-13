"""
Author: Jasper Bussemaker <jasper.bussemaker@dlr.de>

This package is distributed under New BSD license.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from smt.sampling_methods import LHS

try:
    from ConfigSpace import (
        CategoricalHyperparameter,
        Configuration,
        ConfigurationSpace,
        EqualsCondition,
        ForbiddenAndConjunction,
        ForbiddenEqualsClause,
        ForbiddenInClause,
        InCondition,
        OrdinalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
        ForbiddenLessThanRelation,
    )
    from ConfigSpace.exceptions import ForbiddenValueError
    from ConfigSpace.util import get_random_neighbor

    HAS_CONFIG_SPACE = True

except ImportError:
    HAS_CONFIG_SPACE = False

    class Configuration:
        pass

    class ConfigurationSpace:
        pass

    class UniformIntegerHyperparameter:
        pass


def ensure_design_space(xt=None, xlimits=None, design_space=None) -> "BaseDesignSpace":
    """Interface to turn legacy input formats into a DesignSpace"""

    if design_space is not None and isinstance(design_space, BaseDesignSpace):
        return design_space

    if xlimits is not None:
        return DesignSpace(xlimits)

    if xt is not None:
        return DesignSpace([[np.min(xt) - 0.99, np.max(xt) + 1e-4]] * xt.shape[1])

    raise ValueError("Nothing defined that could be interpreted as a design space!")


class DesignVariable:
    """Base class for defining a design variable"""

    upper: Union[float, int]
    lower: Union[float, int]

    def get_typename(self):
        return self.__class__.__name__

    def get_limits(self) -> Union[list, tuple]:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class FloatVariable(DesignVariable):
    """A continuous design variable, varying between its lower and upper bounds"""

    def __init__(self, lower: float, upper: float):
        if upper <= lower:
            raise ValueError(
                f"Upper bound should be higher than lower bound: {upper} <= {lower}"
            )
        self.lower = lower
        self.upper = upper

    def get_limits(self) -> Tuple[float, float]:
        return self.lower, self.upper

    def __str__(self):
        return f"Float ({self.lower}, {self.upper})"

    def __repr__(self):
        return f"{self.get_typename()}({self.lower}, {self.upper})"


class IntegerVariable(DesignVariable):
    """An integer variable that can take any integer value between the bounds (inclusive)"""

    def __init__(self, lower: int, upper: int):
        if upper <= lower:
            raise ValueError(
                f"Upper bound should be higher than lower bound: {upper} <= {lower}"
            )
        self.lower = lower
        self.upper = upper

    def get_limits(self) -> Tuple[int, int]:
        return self.lower, self.upper

    def __str__(self):
        return f"Int ({self.lower}, {self.upper})"

    def __repr__(self):
        return f"{self.get_typename()}({self.lower}, {self.upper})"


class OrdinalVariable(DesignVariable):
    """An ordinal variable that can take any of the given value, and where order between the values matters"""

    def __init__(self, values: List[Union[str, int, float]]):
        if len(values) < 2:
            raise ValueError(f"There should at least be 2 values: {values}")
        self.values = values

    @property
    def lower(self) -> int:
        return 0

    @property
    def upper(self) -> int:
        return len(self.values) - 1

    def get_limits(self) -> List[str]:
        # We convert to integer strings for compatibility reasons
        return [str(i) for i in range(len(self.values))]

    def __str__(self):
        return f"Ord {self.values}"

    def __repr__(self):
        return f"{self.get_typename()}({self.values})"


class CategoricalVariable(DesignVariable):
    """A categorical variable that can take any of the given values, and where order does not matter"""

    def __init__(self, values: List[Union[str, int, float]]):
        if len(values) < 2:
            raise ValueError(f"There should at least be 2 values: {values}")
        self.values = values

    @property
    def lower(self) -> int:
        return 0

    @property
    def upper(self) -> int:
        return len(self.values) - 1

    @property
    def n_values(self):
        return len(self.values)

    def get_limits(self) -> List[Union[str, int, float]]:
        # We convert to strings for compatibility reasons
        return [str(value) for value in self.values]

    def __str__(self):
        return f"Cat {self.values}"

    def __repr__(self):
        return f"{self.get_typename()}({self.values})"


class BaseDesignSpace:
    """
    Interface for specifying (hierarchical) design spaces.

    This class itself only specifies the functionality that any design space definition should implement:
    - a way to specify the design variables, their types, and their bounds or options
    - a way to correct a set of design vectors such that they satisfy all design space hierarchy constraints
    - a way to query which design variables are acting for a set of design vectors
    - a way to impute a set of design vectors such that non-acting design variables are assigned some default value
    - a way to sample n valid design vectors from the design space

    If you want to actually define a design space, use the `DesignSpace` class!

    Note that the correction, querying, and imputation mechanisms should all be implemented in one function
    (`correct_get_acting`), as usually these operations are tightly related.
    """

    def __init__(self, design_variables: List[DesignVariable] = None):
        self._design_variables = design_variables
        self._is_cat_mask = None
        self._is_conditionally_acting_mask = None
        self.seed = None
        self.has_valcons_ord_int = False

    @property
    def design_variables(self) -> List[DesignVariable]:
        if self._design_variables is None:
            self._design_variables = dvs = self._get_design_variables()
            if dvs is None:
                raise RuntimeError(
                    "Design space should either specify the design variables upon initialization "
                    "or as output from _get_design_variables!"
                )
        return self._design_variables

    @property
    def is_cat_mask(self) -> np.ndarray:
        """Boolean mask specifying for each design variable whether it is a categorical variable"""
        if self._is_cat_mask is None:
            self._is_cat_mask = np.array(
                [isinstance(dv, CategoricalVariable) for dv in self.design_variables]
            )
        return self._is_cat_mask

    @property
    def is_all_cont(self) -> bool:
        """Whether or not the space is continuous"""
        is_continuous = all(
            isinstance(dv, FloatVariable) for dv in self.design_variables
        )
        return is_continuous

    @property
    def is_conditionally_acting(self) -> np.ndarray:
        """Boolean mask specifying for each design variable whether it is conditionally acting (can be non-acting)"""
        if self._is_conditionally_acting_mask is None:
            self._is_conditionally_acting_mask = self._is_conditionally_acting()
        return self._is_conditionally_acting_mask

    @property
    def n_dv(self) -> int:
        """Get the number of design variables"""
        return len(self.design_variables)

    def correct_get_acting(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct the given matrix of design vectors and return the corrected vectors and the is_acting matrix.
        It is automatically detected whether input is provided in unfolded space or not.

        Parameters
        ----------
        x: np.ndarray [n_obs, dim]
           - Input variables

        Returns
        -------
        x_corrected: np.ndarray [n_obs, dim]
           - Corrected and imputed input variables
        is_acting: np.ndarray [n_obs, dim]
           - Boolean matrix specifying for each variable whether it is acting or non-acting
        """

        # Detect whether input is provided in unfolded space
        x = np.atleast_2d(x)
        if x.shape[1] == self.n_dv:
            x_is_unfolded = False
        elif x.shape[1] == self._get_n_dim_unfolded():
            x_is_unfolded = True
        else:
            raise ValueError(f"Incorrect shape, expecting {self.n_dv} columns!")

        # If needed, fold before correcting
        if x_is_unfolded:
            x, _ = self.fold_x(x)

        indi = 0
        for i in self.design_variables:
            if not (isinstance(i, FloatVariable)):
                x[:, indi] = np.int64(np.round(x[:, indi], 0))
            indi += 1

        # Correct and get the is_acting matrix
        x_corrected, is_acting = self._correct_get_acting(x)

        # Check conditionally-acting status
        if np.any(~is_acting[:, ~self.is_conditionally_acting]):
            raise RuntimeError("Unconditionally acting variables cannot be non-acting!")

        # Unfold if needed
        if x_is_unfolded:
            x_corrected, is_acting = self.unfold_x(x_corrected, is_acting)

        return x_corrected, is_acting

    def decode_values(
        self, x: np.ndarray, i_dv: int = None
    ) -> List[Union[str, int, float, list]]:
        """
        Return decoded values: converts ordinal and categorical back to their original values.

        If i_dv is given, decoding is done for one specific design variable only.
        If i_dv=None, decoding will be done for all design variables: 1d input is interpreted as a design vector,
        2d input is interpreted as a set of design vectors.
        """

        def _decode_dv(x_encoded: np.ndarray, i_dv_decode):
            dv = self.design_variables[i_dv_decode]
            if isinstance(dv, (OrdinalVariable, CategoricalVariable)):
                values = dv.values
                decoded_values = [values[int(x_ij)] for x_ij in x_encoded]
                return decoded_values

            # No need to decode integer or float variables
            return list(x_encoded)

        # Decode one design variable
        if i_dv is not None:
            if len(x.shape) == 2:
                x_i = x[:, i_dv]
            elif len(x.shape) == 1:
                x_i = x
            else:
                raise ValueError("Expected either 1 or 2-dimensional matrix!")

            # No need to decode for integer or float variable
            return _decode_dv(x_i, i_dv_decode=i_dv)

        # Decode design vectors
        n_dv = self.n_dv
        is_1d = len(x.shape) == 1
        x_mat = np.atleast_2d(x)
        if x_mat.shape[1] != n_dv:
            raise ValueError(
                f"Incorrect number of inputs, expected {n_dv} design variables, received {x_mat.shape[1]}"
            )

        decoded_des_vars = [_decode_dv(x_mat[:, i], i_dv_decode=i) for i in range(n_dv)]
        decoded_des_vectors = [
            [decoded_des_vars[i][ix] for i in range(n_dv)]
            for ix in range(x_mat.shape[0])
        ]
        return decoded_des_vectors[0] if is_1d else decoded_des_vectors

    def sample_valid_x(
        self, n: int, unfolded=False, random_state=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n design vectors and additionally return the is_acting matrix.

        Parameters
        ----------
        n: int
           - Number of samples to generate
        unfolded: bool
           - Whether to return the samples in unfolded space (each categorical level gets its own dimension)

        Returns
        -------
        x: np.ndarray [n, dim]
           - Valid design vectors
        is_acting: np.ndarray [n, dim]
           - Boolean matrix specifying for each variable whether it is acting or non-acting
        """

        # Sample from the design space

        x, is_acting = self._sample_valid_x(n, random_state=random_state)

        # Check conditionally-acting status
        if np.any(~is_acting[:, ~self.is_conditionally_acting]):
            raise RuntimeError("Unconditionally acting variables cannot be non-acting!")

        # Unfold if needed
        if unfolded:
            x, is_acting = self.unfold_x(x, is_acting)

        return x, is_acting

    def get_x_limits(self) -> list:
        """Returns the variable limit definitions in SMT < 2.0 style"""
        return [dv.get_limits() for dv in self.design_variables]

    def get_num_bounds(self):
        """
        Get bounds for the design space.

        Returns
        -------
        np.ndarray [nx, 2]
           - Bounds of each dimension
        """
        return np.array([(dv.lower, dv.upper) for dv in self.design_variables])

    def get_unfolded_num_bounds(self):
        """
        Get bounds for the unfolded continuous space.

        Returns
        -------
        np.ndarray [nx cont, 2]
           - Bounds of each dimension where limits for categorical variables are expanded to [0, 1]
        """
        unfolded_x_limits = []
        for dv in self.design_variables:
            if isinstance(dv, CategoricalVariable):
                unfolded_x_limits += [[0, 1]] * dv.n_values

            elif isinstance(dv, OrdinalVariable):
                # Note that this interpretation is slightly different from the original mixed_integer implementation in
                # smt: we simply map ordinal values to integers, instead of converting them to integer literals
                # This ensures that each ordinal value gets sampled evenly, also if the values themselves represent
                # unevenly spaced (e.g. log-spaced) values
                unfolded_x_limits.append([dv.lower, dv.upper])

            else:
                unfolded_x_limits.append(dv.get_limits())

        return np.array(unfolded_x_limits).astype(float)

    def fold_x(
        self,
        x: np.ndarray,
        is_acting: np.ndarray = None,
        fold_mask: np.ndarray = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fold x and optionally is_acting. Folding reverses the one-hot encoding of categorical variables applied by
        unfolding.

        Parameters
        ----------
        x: np.ndarray [n, dim_unfolded]
           - Unfolded samples
        is_acting: np.ndarray [n, dim_unfolded]
           - Boolean matrix specifying for each unfolded variable whether it is acting or non-acting
        fold_mask: np.ndarray [dim_folded]
           - Mask specifying which design variables to apply folding for

        Returns
        -------
        x_folded: np.ndarray [n, dim]
           - Folded samples
        is_acting_folded: np.ndarray [n, dim]
           - (Optional) boolean matrix specifying for each folded variable whether it is acting or non-acting
        """

        # Get number of unfolded dimension
        x = np.atleast_2d(x)
        x_folded = np.zeros((x.shape[0], len(self.design_variables)))
        is_acting_folded = (
            np.ones(x_folded.shape, dtype=bool) if is_acting is not None else None
        )

        i_x_unfold = 0
        for i, dv in enumerate(self.design_variables):
            if (isinstance(dv, CategoricalVariable)) and (
                fold_mask is None or fold_mask[i]
            ):
                n_dim_cat = dv.n_values

                # Categorical values are folded by reversed one-hot encoding:
                # [[1, 0, 0], [0, 1, 0], [0, 0, 1]] --> [0, 1, 2].T
                x_cat_unfolded = x[:, i_x_unfold : i_x_unfold + n_dim_cat]
                value_index = np.argmax(x_cat_unfolded, axis=1)
                x_folded[:, i] = value_index

                # The is_acting matrix is repeated column-wise, so we can just take the first column
                if is_acting is not None:
                    is_acting_folded[:, i] = is_acting[:, i_x_unfold]

                i_x_unfold += n_dim_cat

            else:
                x_folded[:, i] = x[:, i_x_unfold]
                if is_acting is not None:
                    is_acting_folded[:, i] = is_acting[:, i_x_unfold]
                i_x_unfold += 1

        return x_folded, is_acting_folded

    def unfold_x(
        self, x: np.ndarray, is_acting: np.ndarray = None, fold_mask: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Unfold x and optionally is_acting. Unfolding creates one extra dimension for each categorical variable using
        one-hot encoding.

        Parameters
        ----------
        x: np.ndarray [n, dim]
           - Folded samples
        is_acting: np.ndarray [n, dim]
           - Boolean matrix specifying for each variable whether it is acting or non-acting
        fold_mask: np.ndarray [dim_folded]
           - Mask specifying which design variables to apply folding for

        Returns
        -------
        x_unfolded: np.ndarray [n, dim_unfolded]
           - Unfolded samples
        is_acting_unfolded: np.ndarray [n, dim_unfolded]
           - (Optional) boolean matrix specifying for each unfolded variable whether it is acting or non-acting
        """

        # Get number of unfolded dimension
        n_dim_unfolded = self._get_n_dim_unfolded()
        x = np.atleast_2d(x)
        x_unfolded = np.zeros((x.shape[0], n_dim_unfolded))
        is_acting_unfolded = (
            np.ones(x_unfolded.shape, dtype=bool) if is_acting is not None else None
        )

        i_x_unfold = 0
        for i, dv in enumerate(self.design_variables):
            if isinstance(dv, CategoricalVariable) and (
                fold_mask is None or fold_mask[i]
            ):
                n_dim_cat = dv.n_values
                x_cat = x_unfolded[:, i_x_unfold : i_x_unfold + n_dim_cat]

                # Categorical values are unfolded by one-hot encoding:
                # [0, 1, 2].T --> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                x_i_int = x[:, i].astype(int)
                for i_level in range(n_dim_cat):
                    has_value_mask = x_i_int == i_level
                    x_cat[has_value_mask, i_level] = 1

                # The is_acting matrix is simply repeated column-wise
                if is_acting is not None:
                    is_acting_unfolded[:, i_x_unfold : i_x_unfold + n_dim_cat] = (
                        np.tile(is_acting[:, [i]], (1, n_dim_cat))
                    )

                i_x_unfold += n_dim_cat

            else:
                x_unfolded[:, i_x_unfold] = x[:, i]
                if is_acting is not None:
                    is_acting_unfolded[:, i_x_unfold] = is_acting[:, i]
                i_x_unfold += 1

        x_unfolded = x_unfolded[:, :i_x_unfold]
        if is_acting is not None:
            is_acting_unfolded = is_acting_unfolded[:, :i_x_unfold]

        return x_unfolded, is_acting_unfolded

    def _get_n_dim_unfolded(self) -> int:
        return sum(
            [
                dv.n_values if isinstance(dv, CategoricalVariable) else 1
                for dv in self.design_variables
            ]
        )

    @staticmethod
    def _round_equally_distributed(x_cont, lower: int, upper: int):
        """
        To ensure equal distribution of continuous values to discrete values, we first stretch-out the continuous values
        to extend to 0.5 beyond the integer limits and then round. This ensures that the values at the limits get a
        large-enough share of the continuous values.
        """

        x_cont[x_cont < lower] = lower
        x_cont[x_cont > upper] = upper

        diff = upper - lower
        x_stretched = (x_cont - lower) * ((diff + 0.9999) / (diff + 1e-16)) - 0.5
        return np.round(x_stretched) + lower

    """IMPLEMENT FUNCTIONS BELOW"""

    def _get_design_variables(self) -> List[DesignVariable]:
        """Return the design variables defined in this design space if not provided upon initialization of the class"""

    def _is_conditionally_acting(self) -> np.ndarray:
        """
        Return for each design variable whether it is conditionally acting or not. A design variable is conditionally
        acting if it MAY be non-acting.

        Returns
        -------
        is_conditionally_acting: np.ndarray [dim]
            - Boolean vector specifying for each design variable whether it is conditionally acting
        """
        raise NotImplementedError

    def _correct_get_acting(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Correct the given matrix of design vectors and return the corrected vectors and the is_acting matrix.

        Parameters
        ----------
        x: np.ndarray [n_obs, dim]
           - Input variables

        Returns
        -------
        x_corrected: np.ndarray [n_obs, dim]
           - Corrected and imputed input variables
        is_acting: np.ndarray [n_obs, dim]
           - Boolean matrix specifying for each variable whether it is acting or non-acting
        """
        raise NotImplementedError

    def _sample_valid_x(
        self,
        n: int,
        random_state=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n design vectors and additionally return the is_acting matrix.

        Returns
        -------
        x: np.ndarray [n, dim]
           - Valid design vectors
        is_acting: np.ndarray [n, dim]
           - Boolean matrix specifying for each variable whether it is acting or non-acting
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


VarValueType = Union[int, str, List[Union[int, str]]]


def raise_config_space():
    raise RuntimeError("Dependencies are not installed, run: pip install smt[cs]")


class DesignSpace(BaseDesignSpace):
    """
    Class for defining a (hierarchical) design space by defining design variables, defining decreed variables
    (optional), and adding value constraints (optional).

    Numerical bounds can be requested using `get_num_bounds()`.
    If needed, it is possible to get the legacy SMT < 2.0 `xlimits` format using `get_x_limits()`.

    Parameters
    ----------
    design_variables: list[DesignVariable]
       - The list of design variables: FloatVariable, IntegerVariable, OrdinalVariable, or CategoricalVariable

    Examples
    --------
    Instantiate the design space with all its design variables:

    >>> from smt.utils.design_space import *
    >>> ds = DesignSpace([
    >>>     CategoricalVariable(['A', 'B']),  # x0 categorical: A or B; order is not relevant
    >>>     OrdinalVariable(['C', 'D', 'E']),  # x1 ordinal: C, D or E; order is relevant
    >>>     IntegerVariable(0, 2),  # x2 integer between 0 and 2 (inclusive): 0, 1, 2
    >>>     FloatVariable(0, 1),  # c3 continuous between 0 and 1
    >>> ])
    >>> assert len(ds.design_variables) == 4

    You can define decreed variables (conditional activation):

    >>> ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value='A')  # Activate x1 if x0 == A

    Decreed variables can be chained (however no cycles and no "diamonds" are supported):
    Note: only if ConfigSpace is installed! pip install smt[cs]
    >>> ds.declare_decreed_var(decreed_var=2, meta_var=1, meta_value=['C', 'D'])  # Activate x2 if x1 == C or D

    If combinations of values between two variables are not allowed, this can be done using a value constraint:
    Note: only if ConfigSpace is installed! pip install smt[cs]
    >>> ds.add_value_constraint(var1=0, value1='A', var2=2, value2=[0, 1])  # Forbid x0 == A && x2 == 0 or 1

    After defining everything correctly, you can then use the design space object to correct design vectors and get
    information about which design variables are acting:

    >>> x_corr, is_acting = ds.correct_get_acting(np.array([
    >>>     [0, 0, 2, .25],
    >>>     [0, 2, 1, .75],
    >>> ]))
    >>> assert np.all(x_corr == np.array([
    >>>     [0, 0, 2, .25],
    >>>     [0, 2, 0, .75],
    >>> ]))
    >>> assert np.all(is_acting == np.array([
    >>>     [True, True, True, True],
    >>>     [True, True, False, True],  # x2 is not acting if x1 != C or D (0 or 1)
    >>> ]))

    It is also possible to randomly sample design vectors conforming to the constraints:

    >>> x_sampled, is_acting_sampled = ds.sample_valid_x(100)

    You can also instantiate a purely-continuous design space from bounds directly:

    >>> continuous_design_space = DesignSpace([(0, 1), (0, 2), (.5, 5.5)])
    >>> assert continuous_design_space.n_dv == 3

    If needed, it is possible to get the legacy design space definition format:

    >>> xlimits = ds.get_x_limits()
    >>> cont_bounds = ds.get_num_bounds()
    >>> unfolded_cont_bounds = ds.get_unfolded_num_bounds()

    """

    def __init__(
        self,
        design_variables: Union[List[DesignVariable], list, np.ndarray],
        random_state=None,
    ):
        self.sampler = None

        # Assume float variable bounds as inputs
        def _is_num(val):
            try:
                float(val)
                return True
            except ValueError:
                return False

        if len(design_variables) > 0 and not isinstance(
            design_variables[0], DesignVariable
        ):
            converted_dvs = []
            for bounds in design_variables:
                if len(bounds) != 2 or not _is_num(bounds[0]) or not _is_num(bounds[1]):
                    raise RuntimeError(
                        f"Expecting either a list of DesignVariable objects or float variable "
                        f"bounds! Unrecognized: {bounds!r}"
                    )
                converted_dvs.append(FloatVariable(bounds[0], bounds[1]))
            design_variables = converted_dvs

        self.random_state = random_state  # For testing

        self._cs = None
        self._cs_cate = None
        if HAS_CONFIG_SPACE:
            cs_vars = {}
            cs_vars_cate = {}
            self.isinteger = False
            for i, dv in enumerate(design_variables):
                name = f"x{i}"
                if isinstance(dv, FloatVariable):
                    cs_vars[name] = UniformFloatHyperparameter(
                        name, lower=dv.lower, upper=dv.upper
                    )
                    cs_vars_cate[name] = UniformFloatHyperparameter(
                        name, lower=dv.lower, upper=dv.upper
                    )
                elif isinstance(dv, IntegerVariable):
                    cs_vars[name] = FixedIntegerParam(
                        name, lower=dv.lower, upper=dv.upper
                    )
                    listvalues = []
                    for i in range(int(dv.upper - dv.lower + 1)):
                        listvalues.append(str(int(i + dv.lower)))
                    cs_vars_cate[name] = CategoricalHyperparameter(
                        name, choices=listvalues
                    )
                    self.isinteger = True
                elif isinstance(dv, OrdinalVariable):
                    cs_vars[name] = OrdinalHyperparameter(name, sequence=dv.values)
                    cs_vars_cate[name] = CategoricalHyperparameter(
                        name, choices=dv.values
                    )

                elif isinstance(dv, CategoricalVariable):
                    cs_vars[name] = CategoricalHyperparameter(name, choices=dv.values)
                    cs_vars_cate[name] = CategoricalHyperparameter(
                        name, choices=dv.values
                    )

                else:
                    raise ValueError(f"Unknown variable type: {dv!r}")
            seed = self._to_seed(random_state)

            self._cs = NoDefaultConfigurationSpace(space=cs_vars, seed=seed)
            ## Fix to make constraints work correctly with either IntegerVariable or OrdinalVariable
            ## ConfigSpace is malfunctioning
            self._cs_cate = NoDefaultConfigurationSpace(space=cs_vars_cate, seed=seed)

        # dict[int, dict[any, list[int]]]: {meta_var_idx: {value: [decreed_var_idx, ...], ...}, ...}
        self._meta_vars = {}
        self._is_decreed = np.zeros((len(design_variables),), dtype=bool)

        super().__init__(design_variables)

    def declare_decreed_var(
        self, decreed_var: int, meta_var: int, meta_value: VarValueType
    ):
        """
        Define a conditional (decreed) variable to be active when the meta variable has (one of) the provided values.

        Parameters
        ----------
        decreed_var: int
           - Index of the conditional variable (the variable that is conditionally active)
        meta_var: int
           - Index of the meta variable (the variable that determines whether the conditional var is active)
        meta_value: int | str | list[int|str]
           - The value or list of values that the meta variable can have to activate the decreed var
        """

        # ConfigSpace implementation
        if self._cs is not None:
            # Get associated parameters
            decreed_param = self._get_param(decreed_var)
            meta_param = self._get_param(meta_var)

            # Add a condition that checks for equality (if single value given) or in-collection (if sequence given)
            if isinstance(meta_value, Sequence):
                condition = InCondition(decreed_param, meta_param, meta_value)
            else:
                condition = EqualsCondition(decreed_param, meta_param, meta_value)

            ## Fix to make constraints work correctly with either IntegerVariable or OrdinalVariable
            ## ConfigSpace is malfunctioning
            self._cs.add_condition(condition)
            decreed_param = self._get_param2(decreed_var)
            meta_param = self._get_param2(meta_var)
            # Add a condition that checks for equality (if single value given) or in-collection (if sequence given)
            if isinstance(meta_value, Sequence):
                try:
                    condition = InCondition(
                        decreed_param,
                        meta_param,
                        list(np.atleast_1d(np.array(meta_value, dtype=str))),
                    )
                except ValueError:
                    condition = InCondition(
                        decreed_param,
                        meta_param,
                        list(np.atleast_1d(np.array(meta_value, dtype=float))),
                    )
            else:
                try:
                    condition = EqualsCondition(
                        decreed_param, meta_param, str(meta_value)
                    )
                except ValueError:
                    condition = EqualsCondition(decreed_param, meta_param, meta_value)

            self._cs_cate.add_condition(condition)

        # Simplified implementation
        else:
            # Variables cannot be both meta and decreed at the same time
            if self._is_decreed[meta_var]:
                raise RuntimeError(
                    f"Variable cannot be both meta and decreed ({meta_var})!"
                )

            # Variables can only be decreed by one meta var
            if self._is_decreed[decreed_var]:
                raise RuntimeError(f"Variable is already decreed: {decreed_var}")

            # Define meta-decreed relationship
            if meta_var not in self._meta_vars:
                self._meta_vars[meta_var] = {}

            meta_var_obj = self.design_variables[meta_var]
            for value in (
                meta_value if isinstance(meta_value, Sequence) else [meta_value]
            ):
                encoded_value = value
                if isinstance(meta_var_obj, (OrdinalVariable, CategoricalVariable)):
                    if value in meta_var_obj.values:
                        encoded_value = meta_var_obj.values.index(value)

                if encoded_value not in self._meta_vars[meta_var]:
                    self._meta_vars[meta_var][encoded_value] = []
                self._meta_vars[meta_var][encoded_value].append(decreed_var)

        # Mark as decreed (conditionally acting)
        self._is_decreed[decreed_var] = True

    def add_value_constraint(
        self, var1: int, value1: VarValueType, var2: int, value2: VarValueType
    ):
        """
        Define a constraint where two variables cannot have the given values at the same time.

        Parameters
        ----------
        var1: int
           - Index of the first variable
        value1: int | str | list[int|str]
           - Value or values that the first variable is checked against
        var2: int
           - Index of the second variable
        value2: int | str | list[int|str]
           - Value or values that the second variable is checked against
        """
        if self._cs is None:
            raise_config_space()
        # Get parameters
        param1 = self._get_param(var1)
        param2 = self._get_param(var2)
        mixint_types = (UniformIntegerHyperparameter, OrdinalHyperparameter)
        self.has_valcons_ord_int = isinstance(param1, mixint_types) or isinstance(
            param2, mixint_types
        )
        if not (isinstance(param1, UniformFloatHyperparameter)) and not (
            isinstance(param2, UniformFloatHyperparameter)
        ):
            # Add forbidden clauses
            if isinstance(value1, Sequence):
                clause1 = ForbiddenInClause(param1, value1)
            else:
                clause1 = ForbiddenEqualsClause(param1, value1)

            if isinstance(value2, Sequence):
                clause2 = ForbiddenInClause(param2, value2)
            else:
                clause2 = ForbiddenEqualsClause(param2, value2)

            constraint_clause = ForbiddenAndConjunction(clause1, clause2)
            self._cs.add_forbidden_clause(constraint_clause)
        else:
            if value1 in [">", "<"] and value2 in [">", "<"] and value1 != value2:
                if value1 == "<":
                    constraint_clause = ForbiddenLessThanRelation(param1, param2)
                    self._cs.add_forbidden_clause(constraint_clause)
                else:
                    constraint_clause = ForbiddenLessThanRelation(param2, param1)
                    self._cs.add_forbidden_clause(constraint_clause)
            else:
                raise ValueError("Bad definition of DesignSpace.")

        ## Fix to make constraints work correctly with either IntegerVariable or OrdinalVariable
        ## ConfigSpace is malfunctioning
        # Get parameters
        param1 = self._get_param2(var1)
        param2 = self._get_param2(var2)
        # Add forbidden clauses
        if not (isinstance(param1, UniformFloatHyperparameter)) and not (
            isinstance(param2, UniformFloatHyperparameter)
        ):
            if isinstance(value1, Sequence):
                clause1 = ForbiddenInClause(
                    param1, list(np.atleast_1d(np.array(value1, dtype=str)))
                )
            else:
                clause1 = ForbiddenEqualsClause(param1, str(value1))

            if isinstance(value2, Sequence):
                try:
                    clause2 = ForbiddenInClause(
                        param2, list(np.atleast_1d(np.array(value2, dtype=str)))
                    )
                except ValueError:
                    clause2 = ForbiddenInClause(
                        param2, list(np.atleast_1d(np.array(value2, dtype=float)))
                    )
            else:
                try:
                    clause2 = ForbiddenEqualsClause(param2, str(value2))
                except ValueError:
                    clause2 = ForbiddenEqualsClause(param2, value2)

            constraint_clause = ForbiddenAndConjunction(clause1, clause2)
            self._cs_cate.add_forbidden_clause(constraint_clause)

    def _get_param(self, idx):
        try:
            return self._cs.get_hyperparameter(f"x{idx}")
        except KeyError:
            raise KeyError(f"Variable not found: {idx}")

    def _get_param2(self, idx):
        try:
            return self._cs_cate.get_hyperparameter(f"x{idx}")
        except KeyError:
            raise KeyError(f"Variable not found: {idx}")

    @property
    def _cs_var_idx(self):
        """
        ConfigurationSpace applies topological sort when adding conditions, so compared to what we expect the order of
        parameters might have changed.

        This property contains the indices of the params in the ConfigurationSpace.
        """
        names = self._cs.get_hyperparameter_names()
        return np.array(
            [names.index(f"x{ix}") for ix in range(len(self.design_variables))]
        )

    @property
    def _inv_cs_var_idx(self):
        """
        See _cs_var_idx. This function returns the opposite mapping: the positions of our design variables for each
        param.
        """
        return np.array(
            [int(param[1:]) for param in self._cs.get_hyperparameter_names()]
        )

    def _is_conditionally_acting(self) -> np.ndarray:
        # Decreed variables are the conditionally acting variables
        return self._is_decreed

    def _correct_get_acting(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Correct and impute design vectors"""
        x = x.astype(float)
        if self._cs is not None:
            # Normalize value according to what ConfigSpace expects
            self._normalize_x(x)

            # Get corrected Configuration objects by mapping our design vectors
            # to the ordering of the ConfigurationSpace
            inv_cs_var_idx = self._inv_cs_var_idx
            configs = []
            for xi in x:
                configs.append(self._get_correct_config(xi[inv_cs_var_idx]))

            # Convert Configuration objects to design vectors and get the is_active matrix
            return self._configs_to_x(configs)

        # Simplified implementation
        # Correct discrete variables
        x_corr = x.copy()
        self._normalize_x(x_corr, cs_normalize=False)

        # Determine which variables are acting
        is_acting = np.ones(x_corr.shape, dtype=bool)
        is_acting[:, self._is_decreed] = False
        for i, xi in enumerate(x_corr):
            for i_meta, decrees in self._meta_vars.items():
                meta_var_value = xi[i_meta]
                if meta_var_value in decrees:
                    i_decreed_vars = decrees[meta_var_value]
                    is_acting[i, i_decreed_vars] = True

        # Impute non-acting variables
        self._impute_non_acting(x_corr, is_acting)

        return x_corr, is_acting

    def _to_seed(self, random_state=None):
        seed = None
        if isinstance(random_state, int):
            seed = random_state
        elif isinstance(random_state, np.random.RandomState):
            seed = random_state.get_state()[1][0]
        return seed

    def _sample_valid_x(
        self, n: int, random_state=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample design vectors"""
        # Simplified implementation: sample design vectors in unfolded space
        x_limits_unfolded = self.get_unfolded_num_bounds()
        if self.random_state is None:
            self.random_state = random_state

        if self._cs is not None:
            # Sample Configuration objects
            if self.seed is None:
                seed = self._to_seed(random_state)
                self.seed = seed
            self._cs.seed(self.seed)
            if self.seed is not None:
                self.seed += 1
            configs = self._cs.sample_configuration(n)
            if n == 1:
                configs = [configs]
            # Convert Configuration objects to design vectors and get the is_active matrix
            return self._configs_to_x(configs)

        else:
            if self.sampler is None:
                self.sampler = LHS(
                    xlimits=x_limits_unfolded,
                    random_state=random_state,
                    criterion="ese",
                )
            x = self.sampler(n)
            # Fold and cast to discrete
            x, _ = self.fold_x(x)
            self._normalize_x(x, cs_normalize=False)
            # Get acting information and impute
            return self.correct_get_acting(x)

    def _get_correct_config(self, vector: np.ndarray) -> Configuration:
        config = Configuration(self._cs, vector=vector)

        # Unfortunately we cannot directly ask which parameters SHOULD be active
        # https://github.com/automl/ConfigSpace/issues/253#issuecomment-1513216665
        # Therefore, we temporarily fix it with a very dirty workaround: catch the error raised in check_configuration
        # to find out which parameters should be inactive
        while True:
            try:
                ## Fix to make constraints work correctly with either IntegerVariable or OrdinalVariable
                ## ConfigSpace is malfunctioning
                if self.isinteger and self.has_valcons_ord_int:
                    vector2 = np.copy(vector)
                    self._cs_denormalize_x_ordered(np.atleast_2d(vector2))
                    indvec = 0
                    for hp in self._cs_cate:
                        if (
                            (str(self._cs.get_hyperparameter(hp)).split()[2])
                            == "UniformInteger,"
                            and (
                                str(self._cs_cate.get_hyperparameter(hp)).split()[2][:3]
                            )
                            == "Cat"
                            and not (np.isnan(vector2[indvec]))
                        ):
                            vector2[indvec] = int(vector2[indvec]) - int(
                                str(self._cs_cate.get_hyperparameter(hp)).split()[4][
                                    1:-1
                                ]
                            )
                        indvec += 1
                    self._normalize_x_no_integer(np.atleast_2d(vector2))
                    config2 = Configuration(self._cs_cate, vector=vector2)
                    config2.is_valid_configuration()

                config.is_valid_configuration()
                return config

            except ValueError as e:
                error_str = str(e)
                if "Inactive hyperparameter" in error_str:
                    # Deduce which parameter is inactive
                    inactive_param_name = error_str.split("'")[1]
                    param_idx = self._cs.get_idx_by_hyperparameter_name(
                        inactive_param_name
                    )

                    # Modify the vector and create a new Configuration
                    vector = config.get_array().copy()
                    vector[param_idx] = np.nan
                    config = Configuration(self._cs, vector=vector)

                # At this point, the parameter active statuses are set correctly, so we only need to correct the
                # configuration to one that does not violate the forbidden clauses
                elif isinstance(e, ForbiddenValueError):
                    if self.seed is None:
                        seed = self._to_seed(self.random_state)
                        self.seed = seed
                    if not (self.has_valcons_ord_int):
                        return get_random_neighbor(config, seed=self.seed)
                    else:
                        vector = config.get_array().copy()
                        indvec = 0
                        vector2 = np.copy(vector)
                        ## Fix to make constraints work correctly with either IntegerVariable or OrdinalVariable
                        ## ConfigSpace is malfunctioning
                        for hp in self._cs_cate:
                            if (
                                str(self._cs_cate.get_hyperparameter(hp)).split()[2][:3]
                            ) == "Cat" and not (np.isnan(vector2[indvec])):
                                vector2[indvec] = int(vector2[indvec])
                            indvec += 1

                        config2 = Configuration(self._cs_cate, vector=vector2)
                        config3 = get_random_neighbor(config2, seed=self.seed)
                        vector3 = config3.get_array().copy()
                        config4 = Configuration(self._cs, vector=vector3)
                        return config4
                else:
                    raise

    def _configs_to_x(
        self, configs: List["Configuration"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.zeros((len(configs), len(self.design_variables)))
        is_acting = np.zeros(x.shape, dtype=bool)
        if len(configs) == 0:
            return x, is_acting

        cs_var_idx = self._cs_var_idx
        for i, config in enumerate(configs):
            x[i, :] = config.get_array()[cs_var_idx]

        # De-normalize continuous and integer variables
        self._cs_denormalize_x(x)

        # Set is_active flags and impute x
        is_acting = np.isfinite(x)
        self._impute_non_acting(x, is_acting)

        return x, is_acting

    def _impute_non_acting(self, x: np.ndarray, is_acting: np.ndarray):
        for i, dv in enumerate(self.design_variables):
            if isinstance(dv, FloatVariable):
                # Impute continuous variables to the mid of their bounds
                x[~is_acting[:, i], i] = 0.5 * (dv.upper - dv.lower)

            else:
                # Impute discrete variables to their lower bounds
                lower = 0
                if isinstance(dv, (IntegerVariable, OrdinalVariable)):
                    lower = dv.lower

                x[~is_acting[:, i], i] = lower

    def _normalize_x(self, x: np.ndarray, cs_normalize=True):
        for i, dv in enumerate(self.design_variables):
            if isinstance(dv, FloatVariable):
                if cs_normalize:
                    dv.lower = min(np.min(x[:, i]), dv.lower)
                    dv.upper = max(np.max(x[:, i]), dv.upper)
                    x[:, i] = np.clip(
                        (x[:, i] - dv.lower) / (dv.upper - dv.lower + 1e-16), 0, 1
                    )

            elif isinstance(dv, IntegerVariable):
                x[:, i] = self._round_equally_distributed(x[:, i], dv.lower, dv.upper)

                if cs_normalize:
                    # After rounding, normalize between 0 and 1, where 0 and 1 represent the stretched bounds
                    x[:, i] = (x[:, i] - dv.lower + 0.49999) / (
                        dv.upper - dv.lower + 0.9999
                    )

    def _normalize_x_no_integer(self, x: np.ndarray, cs_normalize=True):
        ordereddesign_variables = [
            self.design_variables[i] for i in self._inv_cs_var_idx
        ]
        for i, dv in enumerate(ordereddesign_variables):
            if isinstance(dv, FloatVariable):
                if cs_normalize:
                    x[:, i] = np.clip(
                        (x[:, i] - dv.lower) / (dv.upper - dv.lower + 1e-16), 0, 1
                    )

            elif isinstance(dv, (OrdinalVariable, CategoricalVariable)):
                # To ensure equal distribution of continuous values to discrete values, we first stretch-out the
                # continuous values to extend to 0.5 beyond the integer limits and then round. This ensures that the
                # values at the limits get a large-enough share of the continuous values
                x[:, i] = self._round_equally_distributed(x[:, i], dv.lower, dv.upper)

    def _cs_denormalize_x(self, x: np.ndarray):
        for i, dv in enumerate(self.design_variables):
            if isinstance(dv, FloatVariable):
                x[:, i] = x[:, i] * (dv.upper - dv.lower) + dv.lower

            elif isinstance(dv, IntegerVariable):
                # Integer values are normalized similarly to what is done in _round_equally_distributed
                x[:, i] = np.round(
                    x[:, i] * (dv.upper - dv.lower + 0.9999) + dv.lower - 0.49999
                )

    def _cs_denormalize_x_ordered(self, x: np.ndarray):
        ordereddesign_variables = [
            self.design_variables[i] for i in self._inv_cs_var_idx
        ]
        for i, dv in enumerate(ordereddesign_variables):
            if isinstance(dv, FloatVariable):
                x[:, i] = x[:, i] * (dv.upper - dv.lower) + dv.lower

            elif isinstance(dv, IntegerVariable):
                # Integer values are normalized similarly to what is done in _round_equally_distributed
                x[:, i] = np.round(
                    x[:, i] * (dv.upper - dv.lower + 0.9999) + dv.lower - 0.49999
                )

    def __str__(self):
        dvs = "\n".join([f"x{i}: {dv!s}" for i, dv in enumerate(self.design_variables)])
        return f"Design space:\n{dvs}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.design_variables!r})"


class NoDefaultConfigurationSpace(ConfigurationSpace):
    """ConfigurationSpace that supports no default configuration"""

    def get_default_configuration(self, *args, **kwargs):
        raise NotImplementedError

    def _check_default_configuration(self, *args, **kwargs):
        pass


class FixedIntegerParam(UniformIntegerHyperparameter):
    def get_neighbors(
        self,
        value: float,
        rs: np.random.RandomState,
        number: int = 4,
        transform: bool = False,
        std: float = 0.2,
    ) -> List[int]:
        # Temporary fix until https://github.com/automl/ConfigSpace/pull/313 is released
        center = self._transform(value)
        lower, upper = self.lower, self.upper
        if upper - lower - 1 < number:
            neighbors = sorted(set(range(lower, upper + 1)) - {center})
            if transform:
                return neighbors
            return self._inverse_transform(np.asarray(neighbors)).tolist()

        return super().get_neighbors(
            value, rs, number=number, transform=transform, std=std
        )
