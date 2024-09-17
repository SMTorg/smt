"""
Author: Jasper Bussemaker <jasper.bussemaker@dlr.de>

This package is distributed under New BSD license.
"""

from typing import List, Optional, Tuple, Union

import numpy as np


try:
    from SMTDesignSpace import *

    HAS_SMTDesignSpace = True

except ImportError:
    HAS_SMTDesignSpace = False


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
    if HAS_ADSG and design_space is not None and isinstance(design_space, ADSG):
        return _convert_adsg_to_legacy(design_space)

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

    def __init__(
        self, design_variables: List[DesignVariable] = None, random_state=None
    ):
        self._design_variables = design_variables
        self._is_cat_mask = None
        self._is_conditionally_acting_mask = None
        self.seed = random_state
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

    def _to_seed(self, random_state=None):
        seed = None
        if isinstance(random_state, int):
            seed = random_state
        elif isinstance(random_state, np.random.RandomState):
            seed = random_state.get_state()[1][0]
        return seed

    def _get_design_variables(self) -> List[DesignVariable]:
        """Return the design variables defined in this design space if not provided upon initialization of the class"""

    """IMPLEMENT FUNCTIONS BELOW"""

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
