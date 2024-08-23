"""
Author: Jasper Bussemaker <jasper.bussemaker@dlr.de>
"""

import contextlib
import itertools
import unittest

import numpy as np

import smt.utils.design_space as ds
from smt.sampling_methods import LHS
from smt.utils.design_space import (
    HAS_CONFIG_SPACE,
    HAS_ADSG,
    ArchDesignSpaceGraph,
    BaseDesignSpace,
    CategoricalVariable,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)


@contextlib.contextmanager
def simulate_no_config_space(do_simulate=True):
    if ds.HAS_CONFIG_SPACE and do_simulate:
        ds.HAS_CONFIG_SPACE = False
        yield
        ds.HAS_CONFIG_SPACE = True
    else:
        yield


class Test(unittest.TestCase):
    def test_design_variables(self):
        with self.assertRaises(ValueError):
            FloatVariable(1, 0)

        float_var = FloatVariable(0, 1)
        self.assertEqual(float_var.lower, 0)
        self.assertEqual(float_var.upper, 1)
        self.assertEqual(float_var.get_limits(), (0, 1))
        self.assertTrue(str(float_var))
        self.assertTrue(repr(float_var))
        self.assertEqual("FloatVariable", float_var.get_typename())

        with self.assertRaises(ValueError):
            IntegerVariable(1, 0)

        int_var = IntegerVariable(0, 1)
        self.assertEqual(int_var.lower, 0)
        self.assertEqual(int_var.upper, 1)
        self.assertEqual(int_var.get_limits(), (0, 1))
        self.assertTrue(str(int_var))
        self.assertTrue(repr(int_var))
        self.assertEqual("IntegerVariable", int_var.get_typename())

        with self.assertRaises(ValueError):
            OrdinalVariable([])
        with self.assertRaises(ValueError):
            OrdinalVariable(["1"])

        ord_var = OrdinalVariable(["A", "B", "C"])
        self.assertEqual(ord_var.values, ["A", "B", "C"])
        self.assertEqual(ord_var.get_limits(), ["0", "1", "2"])
        self.assertEqual(ord_var.lower, 0)
        self.assertEqual(ord_var.upper, 2)
        self.assertTrue(str(ord_var))
        self.assertTrue(repr(ord_var))
        self.assertEqual("OrdinalVariable", ord_var.get_typename())

        with self.assertRaises(ValueError):
            CategoricalVariable([])
        with self.assertRaises(ValueError):
            CategoricalVariable(["A"])

        cat_var = CategoricalVariable(["A", "B", "C"])
        self.assertEqual(cat_var.values, ["A", "B", "C"])
        self.assertEqual(cat_var.get_limits(), ["A", "B", "C"])
        self.assertEqual(cat_var.lower, 0)
        self.assertEqual(cat_var.upper, 2)
        self.assertTrue(str(cat_var))
        self.assertTrue(repr(cat_var))
        self.assertEqual("CategoricalVariable", cat_var.get_typename())

    def test_rounding(self):
        ds = BaseDesignSpace(
            [
                IntegerVariable(0, 5),
                IntegerVariable(-1, 1),
                IntegerVariable(2, 4),
            ]
        )

        x = np.array(
            list(
                itertools.product(
                    np.linspace(0, 5, 20), np.linspace(-1, 1, 20), np.linspace(2, 4, 20)
                )
            )
        )
        for i, dv in enumerate(ds.design_variables):
            self.assertIsInstance(dv, IntegerVariable)
            x[:, i] = ds._round_equally_distributed(x[:, i], dv.lower, dv.upper)

        x1, x1_counts = np.unique(x[:, 0], return_counts=True)
        self.assertTrue(np.all(x1 == [0, 1, 2, 3, 4, 5]))
        x1_counts = x1_counts / np.sum(x1_counts)
        self.assertTrue(np.all(np.abs(x1_counts - np.mean(x1_counts)) <= 0.05))

        x2, x2_counts = np.unique(x[:, 1], return_counts=True)
        self.assertTrue(np.all(x2 == [-1, 0, 1]))
        x2_counts = x2_counts / np.sum(x2_counts)
        self.assertTrue(np.all(np.abs(x2_counts - np.mean(x2_counts)) <= 0.05))

        x3, x3_counts = np.unique(x[:, 2], return_counts=True)
        self.assertTrue(np.all(x3 == [2, 3, 4]))
        x3_counts = x3_counts / np.sum(x3_counts)
        self.assertTrue(np.all(np.abs(x3_counts - np.mean(x3_counts)) <= 0.05))

    def test_base_design_space(self):
        ds = BaseDesignSpace(
            [
                CategoricalVariable(["A", "B"]),
                IntegerVariable(0, 3),
                FloatVariable(-0.5, 0.5),
            ]
        )
        self.assertEqual(ds.get_x_limits(), [["A", "B"], (0, 3), (-0.5, 0.5)])
        self.assertTrue(np.all(ds.get_num_bounds() == [[0, 1], [0, 3], [-0.5, 0.5]]))
        self.assertTrue(
            np.all(
                ds.get_unfolded_num_bounds() == [[0, 1], [0, 1], [0, 3], [-0.5, 0.5]]
            )
        )

        x = np.array(
            [
                [0, 0, 0],
                [1, 2, 0.5],
                [0, 3, 0.5],
            ]
        )
        is_acting = np.array(
            [
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ]
        )

        x_unfolded, is_acting_unfolded = ds.unfold_x(x, is_acting)
        self.assertTrue(
            np.all(
                x_unfolded
                == [
                    [1, 0, 0, 0],
                    [0, 1, 2, 0.5],
                    [1, 0, 3, 0.5],
                ]
            )
        )
        self.assertEqual(is_acting_unfolded.dtype, bool)
        np.testing.assert_array_equal(
            is_acting_unfolded,
            [
                [True, True, True, False],
                [True, True, False, True],
                [False, False, True, True],
            ],
        )

        x_folded, is_acting_folded = ds.fold_x(x_unfolded, is_acting_unfolded)
        np.testing.assert_array_equal(x_folded, x)
        np.testing.assert_array_equal(is_acting_folded, is_acting)

        x_unfold_mask, is_act_unfold_mask = ds.unfold_x(
            x, is_acting, fold_mask=np.array([False] * 3)
        )
        np.testing.assert_array_equal(x_unfold_mask, x)
        np.testing.assert_array_equal(is_act_unfold_mask, is_acting)

        x_fold_mask, is_act_fold_mask = ds.fold_x(
            x, is_acting, fold_mask=np.array([False] * 3)
        )
        np.testing.assert_array_equal(x_fold_mask, x)

        np.testing.assert_array_equal(is_act_fold_mask, is_acting)

    def test_create_design_space(self):
        DesignSpace([FloatVariable(0, 1)])
        with simulate_no_config_space():
            DesignSpace([FloatVariable(0, 1)])

    def test_design_space(self):
        ds = DesignSpace(
            [
                CategoricalVariable(["A", "B", "C"]),
                OrdinalVariable(["0", "1"]),
                IntegerVariable(-1, 2),
                FloatVariable(0.5, 1.5),
            ],
            random_state=42,
        )
        self.assertEqual(len(ds.design_variables), 4)
        if HAS_CONFIG_SPACE:
            self.assertEqual(len(list(ds._cs.values())), 4)
        self.assertTrue(np.all(~ds.is_conditionally_acting))
        if HAS_CONFIG_SPACE:
            x, is_acting = ds.sample_valid_x(3, random_state=42)
            self.assertEqual(x.shape, (3, 4))
            np.testing.assert_allclose(
                x,
                np.array(
                    [
                        [1.0, 0.0, -0.0, 0.83370861],
                        [2.0, 0.0, -1.0, 0.64286682],
                        [2.0, 0.0, -0.0, 1.15088847],
                    ]
                ),
                atol=1e-8,
            )
        else:
            ds.sample_valid_x(3, random_state=42)
            x = np.array(
                [
                    [1, 0, 0, 0.834],
                    [2, 0, -1, 0.6434],
                    [2, 0, 0, 1.151],
                ]
            )
            x, is_acting = ds.correct_get_acting(x)

        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(is_acting.shape, x.shape)

        self.assertEqual(ds.decode_values(x, i_dv=0), ["B", "C", "C"])
        self.assertEqual(ds.decode_values(x, i_dv=1), ["0", "0", "0"])
        self.assertEqual(ds.decode_values(np.array([0, 1, 2]), i_dv=0), ["A", "B", "C"])
        self.assertEqual(ds.decode_values(np.array([0, 1]), i_dv=1), ["0", "1"])

        self.assertEqual(ds.decode_values(x[0, :]), ["B", "0", 0, x[0, 3]])
        self.assertEqual(ds.decode_values(x[[0], :]), [["B", "0", 0, x[0, 3]]])
        self.assertEqual(
            ds.decode_values(x),
            [
                ["B", "0", 0, x[0, 3]],
                ["C", "0", -1, x[1, 3]],
                ["C", "0", 0, x[2, 3]],
            ],
        )

        x_corr, is_act_corr = ds.correct_get_acting(x)
        self.assertTrue(np.all(x_corr == x))
        self.assertTrue(np.all(is_act_corr == is_acting))

        x_sampled_externally = LHS(
            xlimits=ds.get_unfolded_num_bounds(), criterion="ese", random_state=42
        )(3)
        x_corr, is_acting_corr = ds.correct_get_acting(x_sampled_externally)
        x_corr, is_acting_corr = ds.fold_x(x_corr, is_acting_corr)
        np.testing.assert_allclose(
            x_corr,
            np.array(
                [
                    [2.0, 0.0, -1.0, 1.34158548],
                    [0.0, 1.0, -0.0, 0.55199817],
                    [1.0, 1.0, 1.0, 1.15663662],
                ]
            ),
            atol=1e-8,
        )
        self.assertTrue(np.all(is_acting_corr))

        x_unfolded, is_acting_unfolded = ds.sample_valid_x(
            3, unfolded=True, random_state=42
        )
        self.assertEqual(x_unfolded.shape, (3, 6))
        if HAS_CONFIG_SPACE:
            np.testing.assert_allclose(
                x_unfolded,
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0, 2.0, 1.11213215],
                        [0.0, 1.0, 0.0, 1.0, -1.0, 1.09482857],
                        [1.0, 0.0, 0.0, 1.0, -1.0, 0.75061044],
                    ]
                ),
                atol=1e-8,
            )

        self.assertTrue(str(ds))
        self.assertTrue(repr(ds))

        ds.correct_get_acting(np.array([[0, 0, 0, 1.6]]))

    def test_folding_mask(self):
        ds = DesignSpace(
            [
                CategoricalVariable(["A", "B", "C"]),
                CategoricalVariable(["A", "B", "C"]),
            ]
        )
        x = np.array([[1, 2]])
        is_act = np.array([[True, False]])

        self.assertEqual(ds._get_n_dim_unfolded(), 6)

        x_unfolded, is_act_unfolded = ds.unfold_x(x, is_act, np.array([True, False]))
        self.assertTrue(np.all(x_unfolded == np.array([[0, 1, 0, 2]])))
        self.assertTrue(
            np.all(is_act_unfolded == np.array([[True, True, True, False]]))
        )

        x_folded, is_act_folded = ds.fold_x(
            x_unfolded, is_act_unfolded, np.array([True, False])
        )
        self.assertTrue(np.all(x_folded == x))
        self.assertTrue(np.all(is_act_folded == is_act))

    def test_float_design_space(self):
        ds = DesignSpace([(0, 1), (0.5, 2.5), (-0.4, 10)])
        assert ds.n_dv == 3
        assert all(isinstance(dv, FloatVariable) for dv in ds.design_variables)
        assert np.all(ds.get_num_bounds() == np.array([[0, 1], [0.5, 2.5], [-0.4, 10]]))

        ds = DesignSpace([[0, 1], [0.5, 2.5], [-0.4, 10]])
        assert ds.n_dv == 3
        assert all(isinstance(dv, FloatVariable) for dv in ds.design_variables)
        assert np.all(ds.get_num_bounds() == np.array([[0, 1], [0.5, 2.5], [-0.4, 10]]))

        ds = DesignSpace(np.array([[0, 1], [0.5, 2.5], [-0.4, 10]]))
        assert ds.n_dv == 3
        assert all(isinstance(dv, FloatVariable) for dv in ds.design_variables)
        assert np.all(ds.get_num_bounds() == np.array([[0, 1], [0.5, 2.5], [-0.4, 10]]))

    def test_design_space_hierarchical(self):
        ds = DesignSpace(
            [
                CategoricalVariable(["A", "B", "C"]),  # x0
                CategoricalVariable(["E", "F"]),  # x1
                IntegerVariable(0, 1),  # x2
                FloatVariable(0.1, 1),  # x3
            ],
            random_state=42,
        )
        ds.declare_decreed_var(
            decreed_var=3, meta_var=0, meta_value="A"
        )  # Activate x3 if x0 == A

        x_cartesian = np.array(
            list(itertools.product([0, 1, 2], [0, 1], [0, 1], [0.25, 0.75]))
        )
        self.assertEqual(x_cartesian.shape, (24, 4))

        self.assertTrue(
            np.all(ds.is_conditionally_acting == [False, False, False, True])
        )

        x, is_acting = ds.correct_get_acting(x_cartesian)
        _, is_unique = np.unique(x, axis=0, return_index=True)
        self.assertEqual(len(is_unique), 16)
        np.testing.assert_allclose(
            x[is_unique, :],
            np.array(
                [
                    [0, 0, 0, 0.25],
                    [0, 0, 0, 0.75],
                    [0, 0, 1, 0.25],
                    [0, 0, 1, 0.75],
                    [0, 1, 0, 0.25],
                    [0, 1, 0, 0.75],
                    [0, 1, 1, 0.25],
                    [0, 1, 1, 0.75],
                    [1, 0, 0, 0.55],
                    [1, 0, 1, 0.55],
                    [1, 1, 0, 0.55],
                    [1, 1, 1, 0.55],
                    [2, 0, 0, 0.55],
                    [2, 0, 1, 0.55],
                    [2, 1, 0, 0.55],
                    [2, 1, 1, 0.55],
                ]
            ),
        )
        np.testing.assert_array_equal(
            is_acting[is_unique, :],
            np.array(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                ]
            ),
        )

        x_sampled, is_acting_sampled = ds.sample_valid_x(100, random_state=42)
        assert x_sampled.shape == (100, 4)
        x_sampled[is_acting_sampled[:, 3], 3] = np.round(
            x_sampled[is_acting_sampled[:, 3], 3], 4
        )

        x_corr, is_acting_corr = ds.correct_get_acting(x_sampled)
        self.assertTrue(np.sum(np.abs(x_corr - x_sampled)) < 1e-12)
        self.assertTrue(np.all(is_acting_corr == is_acting_sampled))

        seen_x = set()
        seen_is_acting = set()
        for i, xi in enumerate(x_sampled):
            seen_x.add(tuple(xi))
            seen_is_acting.add(tuple(is_acting_sampled[i, :]))
        if HAS_ADSG:
            assert len(seen_x) == 49
        else:
            assert len(seen_x) == 42
        assert len(seen_is_acting) == 2

    @unittest.skipIf(
        not HAS_CONFIG_SPACE, "Hierarchy ConfigSpace dependency not installed"
    )
    def test_design_space_hierarchical_config_space(self):
        ds = DesignSpace(
            [
                CategoricalVariable(["A", "B", "Cc"]),  # x0
                CategoricalVariable(["E", "F"]),  # x1
                IntegerVariable(0, 1),  # x2
                FloatVariable(0, 1),  # x3
            ],
            random_state=42,
        )
        ds.declare_decreed_var(
            decreed_var=3, meta_var=0, meta_value="A"
        )  # Activate x3 if x0 == A
        ds.add_value_constraint(
            var1=0, value1=["Cc"], var2=1, value2="F"
        )  # Prevent a == C and b == F

        x_cartesian = np.array(
            list(itertools.product([0, 1, 2], [0, 1], [0, 1], [0.25, 0.75]))
        )
        self.assertEqual(x_cartesian.shape, (24, 4))

        self.assertTrue(
            np.all(ds.is_conditionally_acting == [False, False, False, True])
        )

        x, is_acting = ds.correct_get_acting(x_cartesian)
        _, is_unique = np.unique(x, axis=0, return_index=True)
        self.assertEqual(len(is_unique), 14)
        np.testing.assert_array_equal(
            x[is_unique, :],
            np.array(
                [
                    [0, 0, 0, 0.25],
                    [0, 0, 0, 0.75],
                    [0, 0, 1, 0.25],
                    [0, 0, 1, 0.75],
                    [0, 1, 0, 0.25],
                    [0, 1, 0, 0.75],
                    [0, 1, 1, 0.25],
                    [0, 1, 1, 0.75],
                    [1, 0, 0, 0.5],
                    [1, 0, 1, 0.5],
                    [1, 1, 0, 0.5],
                    [1, 1, 1, 0.5],
                    [2, 0, 0, 0.5],
                    [2, 0, 1, 0.5],
                ]
            ),
        )
        np.testing.assert_array_equal(
            is_acting[is_unique, :],
            np.array(
                [
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                    [True, True, True, False],
                ]
            ),
        )

        x_sampled, is_acting_sampled = ds.sample_valid_x(100, random_state=42)
        assert x_sampled.shape == (100, 4)
        x_sampled[is_acting_sampled[:, 3], 3] = np.round(
            x_sampled[is_acting_sampled[:, 3], 3]
        )

        x_corr, is_acting_corr = ds.correct_get_acting(x_sampled)
        self.assertTrue(np.all(x_corr == x_sampled))
        self.assertTrue(np.all(is_acting_corr == is_acting_sampled))

        seen_x = set()
        seen_is_acting = set()
        for i, xi in enumerate(x_sampled):
            seen_x.add(tuple(xi))
            seen_is_acting.add(tuple(is_acting_sampled[i, :]))
        assert len(seen_x) == 14
        assert len(seen_is_acting) == 2

    @unittest.skipIf(
        not HAS_CONFIG_SPACE, "Hierarchy ConfigSpace dependency not installed"
    )
    def test_design_space_continuous(self):
        ds = DesignSpace(
            [
                FloatVariable(0, 1),  # x0
                FloatVariable(0, 1),  # x1
                FloatVariable(0, 1),  # x2
            ],
            random_state=42,
        )
        ds.add_value_constraint(
            var1=0, value1="<", var2=1, value2=">"
        )  # Prevent x0 < x1
        ds.add_value_constraint(
            var1=1, value1="<", var2=2, value2=">"
        )  # Prevent x1 < x2

        # correct_get_acting
        x_sampled, is_acting_sampled = ds.sample_valid_x(100, random_state=42)
        self.assertTrue(np.min(x_sampled[:, 0] - x_sampled[:, 1]) > 0)
        self.assertTrue(np.min(x_sampled[:, 1] - x_sampled[:, 2]) > 0)
        ds = DesignSpace(
            [
                IntegerVariable(0, 2),  # x0
                FloatVariable(0, 2),  # x1
                IntegerVariable(0, 2),  # x2
            ],
            random_state=42,
        )
        ds.add_value_constraint(
            var1=0, value1="<", var2=1, value2=">"
        )  # Prevent x0 < x1
        ds.add_value_constraint(
            var1=1, value1="<", var2=2, value2=">"
        )  # Prevent x0 < x1

        # correct_get_acting
        x_sampled, is_acting_sampled = ds.sample_valid_x(100, random_state=42)
        self.assertTrue(np.min(x_sampled[:, 0] - x_sampled[:, 1]) > 0)
        self.assertTrue(np.min(x_sampled[:, 1] - x_sampled[:, 2]) > 0)

    @unittest.skipIf(
        not HAS_CONFIG_SPACE, "Hierarchy ConfigSpace dependency not installed"
    )
    def test_check_conditionally_acting(self):
        class WrongDesignSpace(DesignSpace):
            def _is_conditionally_acting(self) -> np.ndarray:
                return np.zeros((self.n_dv,), dtype=bool)

        for simulate_no_cs in [True, False]:
            with simulate_no_config_space(simulate_no_cs):
                ds = WrongDesignSpace(
                    [
                        CategoricalVariable(["A", "B", "C"]),  # x0
                        CategoricalVariable(["E", "F"]),  # x1
                        IntegerVariable(0, 1),  # x2
                        FloatVariable(0, 1),  # x3
                    ],
                    random_state=42,
                )
                ds.declare_decreed_var(
                    decreed_var=3, meta_var=0, meta_value="A"
                )  # Activate x3 if x0 == A
                self.assertRaises(
                    RuntimeError, lambda: ds.sample_valid_x(10, random_state=42)
                )

    def test_check_conditionally_acting_2(self):
        for simulate_no_cs in [True, False]:
            with simulate_no_config_space(simulate_no_cs):
                ds = DesignSpace(
                    [
                        CategoricalVariable(["A", "B", "C"]),  # x0
                        CategoricalVariable(["E", "F"]),  # x1
                        IntegerVariable(0, 1),  # x2
                        FloatVariable(0, 1),  # x3
                    ],
                    random_state=42,
                )
                ds.declare_decreed_var(
                    decreed_var=0, meta_var=1, meta_value="E"
                )  # Activate x3 if x0 == A

                ds.sample_valid_x(10, random_state=42)

    @unittest.skipIf(
        not HAS_CONFIG_SPACE, "Hierarchy ConfigSpace dependency not installed"
    )
    def test_restrictive_value_constraint_ordinal(self):
        ds = DesignSpace(
            [
                OrdinalVariable(["0", "1", "2"]),
                OrdinalVariable(["0", "1", "2"]),
            ]
        )
        assert list(ds._cs.values())[0].default_value == "0"

        ds.add_value_constraint(var1=0, value1="1", var2=1, value2="1")
        ds.sample_valid_x(100, random_state=42)

        x_cartesian = np.array(list(itertools.product([0, 1, 2], [0, 1, 2])))
        x_cartesian2, _ = ds.correct_get_acting(x_cartesian)
        np.testing.assert_array_equal(
            np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [0, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            ),
            x_cartesian2,
        )

    @unittest.skipIf(
        not HAS_CONFIG_SPACE, "Hierarchy ConfigSpace dependency not installed"
    )
    def test_restrictive_value_constraint_integer(self):
        ds = DesignSpace(
            [
                IntegerVariable(0, 2),
                IntegerVariable(0, 2),
            ]
        )
        assert list(ds._cs.values())[0].default_value == 1

        ds.add_value_constraint(var1=0, value1=1, var2=1, value2=1)
        ds.sample_valid_x(100, random_state=42)

        x_cartesian = np.array(list(itertools.product([0, 1, 2], [0, 1, 2])))
        ds.correct_get_acting(x_cartesian)
        x_cartesian2, _ = ds.correct_get_acting(x_cartesian)
        np.testing.assert_array_equal(
            np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
            ),
            x_cartesian2,
        )

    @unittest.skipIf(
        not HAS_CONFIG_SPACE, "Hierarchy ConfigSpace dependency not installed"
    )
    def test_restrictive_value_constraint_categorical(self):
        ds = DesignSpace(
            [
                CategoricalVariable(["a", "b", "c"]),
                CategoricalVariable(["a", "b", "c"]),
            ]
        )
        assert list(ds._cs.values())[0].default_value == "a"

        ds.add_value_constraint(var1=0, value1="b", var2=1, value2="b")
        ds.sample_valid_x(100, random_state=42)

        x_cartesian = np.array(list(itertools.product([0, 1, 2], [0, 1, 2])))
        ds.correct_get_acting(x_cartesian)
        x_cartesian2, _ = ds.correct_get_acting(x_cartesian)
        np.testing.assert_array_equal(
            np.array(
                [[0, 0], [0, 1], [0, 2], [1, 0], [0, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
            ),
            x_cartesian2,
        )

    @unittest.skipIf(
        not (HAS_CONFIG_SPACE and HAS_ADSG),
        "Architecture Design Space Graph or ConfigSpace not installed.",
    )
    def test_adsg_to_legacy(self):
        from adsg_core import BasicADSG, NamedNode, DesignVariableNode
        from smt.utils.design_space import ensure_design_space
        from adsg_core import GraphProcessor

        # Create the ADSG
        adsg = BasicADSG()
        ndv = 13
        # Create nodes
        n = [NamedNode(f"N{i}") for i in range(ndv)]
        n = [
            NamedNode("MLP"),
            NamedNode("Learning_rate"),
            NamedNode("Activation_function"),
            NamedNode("Optimizer"),
            NamedNode("Decay"),
            NamedNode("Power_update"),
            NamedNode("Average_start"),
            NamedNode("Running_Average_1"),
            NamedNode("Running_Average_2"),
            NamedNode("Numerical_Stability"),
            NamedNode("Nb_layers"),
            NamedNode("Layer_1"),
            NamedNode("Layer_2"),
            NamedNode("Layer_3"),  # NamedNode("Dropout"),
            NamedNode("ASGD"),
            NamedNode("Adam"),
            NamedNode("20...40"),
            NamedNode("40"),
            NamedNode("45"),
            NamedNode("20...40"),
            NamedNode("40"),
            NamedNode("45"),
            NamedNode("20...40"),
            NamedNode("40"),
            NamedNode("45"),
        ]
        adsg.add_node(n[1])
        adsg.add_node(n[2])
        adsg.add_edges(
            [
                (n[3], n[10]),
                (n[14], n[4]),
                (n[14], n[5]),
                (n[14], n[6]),
                (n[15], n[7]),
                (n[15], n[8]),
                (n[15], n[9]),
            ]
        )
        adsg.add_selection_choice("Optimizer_Choice", n[3], [n[14], n[15]])
        adsg.add_selection_choice("#layers", n[10], [n[11], n[12], n[13]])
        a = []
        for i in range(3):
            a.append(NamedNode(str(25 + 5 * i)))
        b = a.copy()
        b.append(n[17])
        b.append(n[18])
        choicel1 = adsg.add_selection_choice("#neurons_1", n[11], b)
        adsg.add_edges([(n[12], choicel1), (n[13], choicel1)])

        a = []
        for i in range(3):
            a.append(NamedNode(str(25 + 5 * i)))
        b = a.copy()
        b.append(n[20])
        b.append(n[21])
        choicel1 = adsg.add_selection_choice("#neurons_2", n[12], b)
        adsg.add_edges([(n[13], choicel1)])

        a = []
        for i in range(3):
            a.append(NamedNode(str(25 + 5 * i)))
        b = a.copy()
        b.append(n[23])
        b.append(n[24])
        choicel1 = adsg.add_selection_choice("#neurons_3", n[13], b)

        adsg.add_incompatibility_constraint([n[15], n[13]])
        adsg.add_incompatibility_constraint([n[14], n[17]])
        adsg.add_incompatibility_constraint([n[14], n[18]])
        adsg.add_incompatibility_constraint([n[14], n[20]])
        adsg.add_incompatibility_constraint([n[14], n[21]])
        adsg.add_incompatibility_constraint([n[14], n[23]])
        adsg.add_incompatibility_constraint([n[14], n[24]])
        start_nodes = set()
        start_nodes.add(n[3])
        start_nodes.add(n[2])
        start_nodes.add(n[1])
        adsg.add_edges(
            [
                (n[1], DesignVariableNode("x0", bounds=(0, 1))),
                (n[4], DesignVariableNode("x1", bounds=(0, 1))),
                (n[5], DesignVariableNode("x2", bounds=(0, 1))),
                (n[6], DesignVariableNode("x3", bounds=(0, 1))),
                (n[7], DesignVariableNode("x4", bounds=(0, 1))),
                (n[8], DesignVariableNode("x5", bounds=(0, 1))),
                (n[9], DesignVariableNode("x6", bounds=(0, 1))),
            ]
        )
        adsg.add_selection_choice(
            "Activation_Choice",
            n[2],
            [NamedNode("ReLU"), NamedNode("Sigmoid"), NamedNode("Tanh")],
        )
        adsg = adsg.set_start_nodes(start_nodes)
        adsg.render()
        gp = GraphProcessor(adsg)
        gp.get_statistics()
        design_space = ensure_design_space(design_space=adsg)
        np.testing.assert_array_equal(
            np.array(
                [
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
            design_space.is_conditionally_acting,
        )
        design_space2 = ArchDesignSpaceGraph(adsg=adsg)
        np.testing.assert_array_equal(
            np.array(
                [
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ),
            design_space2.is_conditionally_acting,
        )


if __name__ == "__main__":
    unittest.main()
