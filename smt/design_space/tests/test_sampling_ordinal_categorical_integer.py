
from smt.design_space import  DesignSpace

from collections import Counter

from smt.applications.mixed_integer import MixedIntegerSamplingMethod
from smt.design_space import DesignSpace, FloatVariable, IntegerVariable, CategoricalVariable, OrdinalVariable
from smt.sampling_methods.lhs import LHS
from smt.sampling_methods.random import Random

from smt_design_space_ext import ConfigSpaceDesignSpaceImpl
from smt_design_space_ext import AdsgDesignSpaceImpl

import unittest

import sys

LEVELS = ["low", "medium", "high", "very_high"]
N_SAMPLES = 1200

class TestMixedIntegerSamplingDistribution(unittest.TestCase):
    def test_LHS_ordinal_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = DesignSpace([OrdinalVariable(["low", "medium", "high", "very_high"])])
        sampler = MixedIntegerSamplingMethod(LHS, design_space, output_in_folded_space=False)
        samples = sampler(N_SAMPLES)
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

    def test_LHS_integer_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = DesignSpace([IntegerVariable(0, 3)])
        sampler = MixedIntegerSamplingMethod(LHS, design_space, output_in_folded_space=False)
        samples = sampler(N_SAMPLES)
        counts = Counter(row[0] for row in samples)
        distrib = dict(sorted(Counter(row[0] for row in samples).items()))
        print(f"\n Distribution:   {distrib}")

        for val in [0.0, 1.0, 2.0, 3.0]:
            self.assertTrue(250 < counts[val] < 350, f"Expected uniform distribution for {val}, got {counts[val]}")

    def test_LHS_categorical_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = DesignSpace([CategoricalVariable(["low", "medium", "high", "very_high"])])
        sampler = MixedIntegerSamplingMethod(LHS, design_space, output_in_folded_space=True)
        samples = sampler(N_SAMPLES)
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

    def test_Random_ordinal_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = DesignSpace([OrdinalVariable(["low", "medium", "high", "very_high"])])
        sampler = MixedIntegerSamplingMethod(Random, design_space, output_in_folded_space=False)
        samples = sampler(N_SAMPLES)
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

    def test_Random_integer_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = DesignSpace([IntegerVariable(0, 3)])
        sampler = MixedIntegerSamplingMethod(Random, design_space, output_in_folded_space=False)
        samples = sampler(N_SAMPLES)
        counts = Counter(row[0] for row in samples)
        distrib = dict(sorted(Counter(row[0] for row in samples).items()))
        print(f"\n Distribution:   {distrib}")

        for val in [0.0, 1.0, 2.0, 3.0]:
            self.assertTrue(250 < counts[val] < 350, f"Expected uniform distribution for {val}, got {counts[val]}")

    def test_Random_categorical_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = DesignSpace([CategoricalVariable(["low", "medium", "high", "very_high"])])
        sampler = MixedIntegerSamplingMethod(Random, design_space, output_in_folded_space=True)
        samples = sampler(N_SAMPLES)
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

    def test_DesignSpace_sample_valid_x_ordinal_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = DesignSpace([OrdinalVariable(["low", "medium", "high", "very_high"])])
        samples = design_space.sample_valid_x(1200)[0]
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

    def test_DesignSpace_sample_valid_x_integer_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = DesignSpace([IntegerVariable(0, 3)])
        samples = design_space.sample_valid_x(1200)[0]
        counts = Counter(row[0] for row in samples)
        distrib = dict(sorted(Counter(row[0] for row in samples).items()))
        print(f"\n Distribution:   {distrib}")

        for val in [0.0, 1.0, 2.0, 3.0]:
            self.assertTrue(250 < counts[val] < 350, f"Expected uniform distribution for {val}, got {counts[val]}")

    def test_DesignSpace_sample_valid_x_categorical_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = DesignSpace([CategoricalVariable(["low", "medium", "high", "very_high"])])
        samples = design_space.sample_valid_x(1200)[0]
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

    def test_AdsgDesignSpaceImpl_sample_valid_x_ordinal_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = AdsgDesignSpaceImpl(design_variables=[OrdinalVariable(["low", "medium", "high", "very_high"])])
        samples = design_space.sample_valid_x(1200)[0]
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

    def test_AdsgDesignSpaceImpl_sample_valid_x_integer_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = AdsgDesignSpaceImpl(design_variables=[IntegerVariable(0, 3)])
        samples = design_space.sample_valid_x(1200)[0]
        counts = Counter(row[0] for row in samples)
        distrib = dict(sorted(Counter(row[0] for row in samples).items()))
        print(f"\n Distribution:   {distrib}")

        for val in [0.0, 1.0, 2.0, 3.0]:
            self.assertTrue(250 < counts[val] < 350, f"Expected uniform distribution for {val}, got {counts[val]}")

    def test_AdsgDesignSpaceImpl_sample_valid_x_categorical_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = AdsgDesignSpaceImpl(design_variables=[CategoricalVariable(["low", "medium", "high", "very_high"])])
        samples = design_space.sample_valid_x(1200)[0]
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

    def test_ConfigSpaceDesignSpaceImpl_sample_valid_x_ordinal_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = ConfigSpaceDesignSpaceImpl(design_variables=[OrdinalVariable(["low", "medium", "high", "very_high"])])
        samples = design_space.sample_valid_x(1200)[0]
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

    def test_ConfigSpaceSpaceImpl_sample_valid_x_integer_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = ConfigSpaceDesignSpaceImpl(design_variables=[IntegerVariable(0, 3)])
        samples = design_space.sample_valid_x(1200)[0]
        counts = Counter(row[0] for row in samples)
        distrib = dict(sorted(Counter(row[0] for row in samples).items()))
        print(f"\n Distribution:   {distrib}")

        for val in [0.0, 1.0, 2.0, 3.0]:
            self.assertTrue(250 < counts[val] < 350, f"Expected uniform distribution for {val}, got {counts[val]}")

    def test_AdsgDesignSpaceImpl_sample_valid_x_categorical_distribution(self):
        current_function_name = sys._getframe().f_code.co_name
        print(f"\n Executing: {current_function_name}\n")
        design_space = ConfigSpaceDesignSpaceImpl(design_variables=[CategoricalVariable(["low", "medium", "high", "very_high"])])
        samples = design_space.sample_valid_x(1200)[0]
        decoded = design_space.decode_values(samples)
        counts = Counter(row[0] for row in decoded)
        distrib = dict(sorted(Counter(row[0] for row in decoded).items()))
        print(f"\n Distribution:   {distrib}")

        for level in ["low", "medium", "high", "very_high"]:
            self.assertTrue(250 < counts[level] < 350, f"Expected uniform distribution for {level}, got {counts[level]}")

if __name__ == '__main__':
    unittest.main()