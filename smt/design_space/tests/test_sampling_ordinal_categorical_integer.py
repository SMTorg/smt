import unittest
from collections import Counter

from smt.applications.mixed_integer import MixedIntegerSamplingMethod
from smt.design_space import (
    CategoricalVariable,
    DesignSpace,
    IntegerVariable,
    OrdinalVariable,
)
from smt.sampling_methods.lhs import LHS
from smt.sampling_methods.random import Random

LEVELS = ["low", "medium", "high", "very_high"]
INTEGER_VALUES = [0.0, 1.0, 2.0, 3.0]
N_SAMPLES = 1200


class TestMixedIntegerSamplingDistribution(unittest.TestCase):
    def _assert_uniform_distribution(
        self, counts, expected_values, lower=250, upper=350
    ):
        distrib = dict(sorted(counts.items()))
        print(f"\n Distribution:   {distrib}")

        for val in expected_values:
            self.assertTrue(
                lower < counts[val] < upper,
                f"Expected uniform distribution for {val}, got {counts[val]}",
            )

    def _check_distribution(self, design_space, samples, variable, expected_values):
        if isinstance(variable, IntegerVariable):
            counts = Counter(row[0] for row in samples)
        else:
            decoded = design_space.decode_values(samples)
            counts = Counter(row[0] for row in decoded)
        self._assert_uniform_distribution(counts, expected_values)

    def _test_mixed_integer(
        self,
        sampling_method,
        variable,
        expected_values,
        output_in_folded_space=False,
        n_samples=N_SAMPLES,
    ):
        design_space = DesignSpace([variable])
        sampler = MixedIntegerSamplingMethod(
            sampling_method, design_space, output_in_folded_space=output_in_folded_space
        )
        samples = sampler(n_samples)
        self._check_distribution(design_space, samples, variable, expected_values)

    def _test_design_space(
        self,
        design_space_class,
        variable,
        expected_values,
        n_samples=N_SAMPLES,
    ):
        design_space = (
            design_space_class([variable])
            if design_space_class is DesignSpace
            else design_space_class(design_variables=[variable])
        )
        samples = design_space.sample_valid_x(n_samples)[0]
        self._check_distribution(design_space, samples, variable, expected_values)

    def _test_sampler_distribution(self, sampling_method):
        cases = [
            ("ordinal", OrdinalVariable(LEVELS), LEVELS),
            ("integer", IntegerVariable(0, 3), INTEGER_VALUES),
            ("categorical", CategoricalVariable(LEVELS), LEVELS),
        ]
        for name, var, expected in cases:
            for output_in_folded_space in [True, False]:
                with self.subTest(
                    variable_type=name,
                    output_in_folded_space=output_in_folded_space,
                ):
                    space_desc = "folded" if output_in_folded_space else "unfolded"
                    print(
                        f"\n Executing: {sampling_method.__name__}_{name}_{space_desc}_distribution\n"
                    )
                    self._test_mixed_integer(
                        sampling_method,
                        var,
                        expected,
                        output_in_folded_space=output_in_folded_space,
                    )

    def _test_design_space_distribution(self, design_space_class):
        cases = [
            ("ordinal", OrdinalVariable(LEVELS), LEVELS),
            ("integer", IntegerVariable(0, 3), INTEGER_VALUES),
            ("categorical", CategoricalVariable(LEVELS), LEVELS),
        ]
        for name, var, expected in cases:
            with self.subTest(variable_type=name):
                print(
                    f"\n Executing: {design_space_class.__name__}_sample_valid_x_{name}_distribution\n"
                )
                self._test_design_space(design_space_class, var, expected)

    def test_LHS_distribution(self):
        self._test_sampler_distribution(LHS)

    def test_Random_distribution(self):
        self._test_sampler_distribution(Random)

    def test_DesignSpace_sample_valid_x_distribution(self):
        self._test_design_space_distribution(DesignSpace)


if __name__ == "__main__":
    unittest.main()
