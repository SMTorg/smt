"""
Author: Jasper Bussemaker <jasper.bussemaker@dlr.de>

This package is distributed under New BSD license.
"""

import numpy as np
from typing import List, Union, Tuple, Sequence, Optional

from smt.sampling_methods import LHS


def ensure_design_space(xt=None, xlimits=None, design_space=None) -> "BaseDesignSpace":
    """Interface to turn legacy input formats into a DesignSpace"""

    if design_space is not None and isinstance(design_space, BaseDesignSpace):
        return design_space

    if xlimits is not None:
        return DesignSpace(xlimits)

    if xt is not None:
        return DesignSpace([[0, 1]] * xt.shape[1])

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

    def sample_valid_x(self, n: int, unfolded=False) -> Tuple[np.ndarray, np.ndarray]:
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
        x, is_acting = self._sample_valid_x(n)

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
        self, x: np.ndarray, is_acting: np.ndarray = None
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
            if isinstance(dv, CategoricalVariable):
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
        self, x: np.ndarray, is_acting: np.ndarray = None
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
            if isinstance(dv, CategoricalVariable):
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
                    is_acting_unfolded[
                        :, i_x_unfold : i_x_unfold + n_dim_cat
                    ] = np.tile(is_acting[:, [i]], (1, n_dim_cat))

                i_x_unfold += n_dim_cat

            else:
                x_unfolded[:, i_x_unfold] = x[:, i]
                if is_acting is not None:
                    is_acting_unfolded[:, i_x_unfold] = is_acting[:, i]
                i_x_unfold += 1

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

    def _sample_valid_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n design vectors and additionally return the is_acting matrix.

        Returns
        ----------
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
    Class for defining a (hierarchical) design space by defining design variables, and defining decreed variables
    (optional).

    Numerical bounds can be requested using `get_num_bounds()`.
    If needed, it is possible to get the legacy SMT < 2.0 `xlimits` format using `get_x_limits()`.

    Parameters
    ----------
    design_variables: list[DesignVariable]
       - The list of design variables: FloatVariable, IntegerVariable, OrdinalVariable, or CategoricalVariable

    Examples
    --------
    Instantiate the design space with all its design variables:

    >>> print("toto")
    >>> from smt.utils.design_space import DesignSpace, FloatVariable, IntegerVariable, OrdinalVariable, CategoricalVariable
    >>> ds = DesignSpace([
    >>>     CategoricalVariable(['A', 'B']),  # x0 categorical: A or B; order is not relevant
    >>>     OrdinalVariable(['C', 'D', 'E']),  # x1 ordinal: C, D or E; order is relevant
    >>>     IntegerVariable(0, 2),  # x2 integer between 0 and 2 (inclusive): 0, 1, 2
    >>>     FloatVariable(0, 1),  # c3 continuous between 0 and 1
    >>> ])
    >>> assert len(ds.design_variables) == 4

    You can define decreed variables (conditional activation):

    >>> ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value='A')  # Activate x1 if x0 == A

    After defining everything correctly, you can then use the design space object to correct design vectors and get
    information about which design variables are acting:

    >>> x_corr, is_acting = ds.correct_get_acting(np.array([
    >>>     [0, 0, 2, .25],
    >>>     [1, 2, 1, .75],
    >>> ]))
    >>> assert np.all(x_corr == np.array([
    >>>     [0, 0, 2, .25],
    >>>     [1, 0, 1, .75],
    >>> ]))
    >>> assert np.all(is_acting == np.array([
    >>>     [True, True, True, True],
    >>>     [True, False, True, True],  # x1 is not acting if x0 != A
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
        self, design_variables: Union[List[DesignVariable], list, np.ndarray], seed=None
    ):
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

        self.seed = seed  # For testing

        self._meta_vars = (
            {}
        )  # dict[int, dict[any, list[int]]]: {meta_var_idx: {value: [decreed_var_idx, ...], ...}, ...}
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
        for value in meta_value if isinstance(meta_value, Sequence) else [meta_value]:
            encoded_value = value
            if isinstance(meta_var_obj, (OrdinalVariable, CategoricalVariable)):
                if value in meta_var_obj.values:
                    encoded_value = meta_var_obj.values.index(value)

            if encoded_value not in self._meta_vars[meta_var]:
                self._meta_vars[meta_var][encoded_value] = []
            self._meta_vars[meta_var][encoded_value].append(decreed_var)

        # Mark as decreed (conditionally acting)
        self._is_decreed[decreed_var] = True

    def _is_conditionally_acting(self) -> np.ndarray:
        # Decreed variables are the conditionally acting variables
        return self._is_decreed

    def _correct_get_acting(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Correct and impute design vectors"""

        # Simplified implementation
        # Correct discrete variables
        x_corr = x.copy()
        self._normalize_x(x_corr)

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

    def _sample_valid_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample design vectors"""

        # Simplified implementation: sample design vectors in unfolded space
        x_limits_unfolded = self.get_unfolded_num_bounds()
        sampler = LHS(xlimits=x_limits_unfolded, random_state=self.seed)
        x = sampler(n)

        # Cast to discrete and fold
        self._normalize_x(x)
        x, _ = self.fold_x(x)

        # Get acting information and impute
        return self.correct_get_acting(x)

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

    def _normalize_x(self, x: np.ndarray):
        for i, dv in enumerate(self.design_variables):
            if isinstance(dv, IntegerVariable):
                x[:, i] = self._round_equally_distributed(x[:, i], dv.lower, dv.upper)

            elif isinstance(dv, (OrdinalVariable, CategoricalVariable)):
                # To ensure equal distribution of continuous values to discrete values, we first stretch-out the
                # continuous values to extend to 0.5 beyond the integer limits and then round. This ensures that the
                # values at the limits get a large-enough share of the continuous values
                x[:, i] = self._round_equally_distributed(x[:, i], dv.lower, dv.upper)

    def __str__(self):
        dvs = "\n".join([f"x{i}: {dv!s}" for i, dv in enumerate(self.design_variables)])
        return f"Design space:\n{dvs}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.design_variables!r})"
