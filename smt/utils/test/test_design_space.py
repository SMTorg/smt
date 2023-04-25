"""
Author: Jasper Bussemaker <jasper.bussemaker@dlr.de>
"""
import unittest
import itertools
import numpy as np
from smt.sampling_methods import LHS
from smt.utils.design_space import XType, FloatVariable, IntegerVariable, OrdinalVariable, CategoricalVariable,\
    BaseDesignSpace, DesignSpace, LegacyDesignSpace


class Test(unittest.TestCase):

    def test_design_variables(self):
        with self.assertRaises(ValueError):
            FloatVariable(1, 0)

        float_var = FloatVariable(0, 1)
        self.assertEqual(float_var.lower, 0)
        self.assertEqual(float_var.upper, 1)
        self.assertEqual(float_var.get_limits(), (0, 1))
        self.assertEqual(float_var.get_type(), XType.FLOAT)
        self.assertTrue(str(float_var))
        self.assertTrue(repr(float_var))

        with self.assertRaises(ValueError):
            IntegerVariable(1, 0)

        int_var = IntegerVariable(0, 1)
        self.assertEqual(int_var.lower, 0)
        self.assertEqual(int_var.upper, 1)
        self.assertEqual(int_var.get_limits(), (0, 1))
        self.assertEqual(int_var.get_type(), XType.ORD)
        self.assertTrue(str(int_var))
        self.assertTrue(repr(int_var))

        with self.assertRaises(ValueError):
            OrdinalVariable([])
        with self.assertRaises(ValueError):
            OrdinalVariable(['1'])

        ord_var = OrdinalVariable(['A', 'B', 'C'])
        self.assertEqual(ord_var.values, ['A', 'B', 'C'])
        self.assertEqual(ord_var.get_limits(), ['0', '1', '2'])
        self.assertEqual(ord_var.get_type(), XType.ORD)
        self.assertEqual(ord_var.lower, 0)
        self.assertEqual(ord_var.upper, 2)
        self.assertTrue(str(ord_var))
        self.assertTrue(repr(ord_var))

        with self.assertRaises(ValueError):
            CategoricalVariable([])
        with self.assertRaises(ValueError):
            CategoricalVariable(['A'])

        cat_var = CategoricalVariable(['A', 'B', 'C'])
        self.assertEqual(cat_var.values, ['A', 'B', 'C'])
        self.assertEqual(cat_var.get_limits(), ['A', 'B', 'C'])
        self.assertEqual(cat_var.get_type(), (XType.ENUM, 3))
        self.assertEqual(cat_var.lower, 0)
        self.assertEqual(cat_var.upper, 2)
        self.assertTrue(str(cat_var))
        self.assertTrue(repr(cat_var))

    def test_rounding(self):
        ds = BaseDesignSpace([
            IntegerVariable(0, 5),
            IntegerVariable(-1, 1),
            IntegerVariable(2, 4),
        ])

        x = np.array(list(itertools.product(np.linspace(0, 5, 20), np.linspace(-1, 1, 20), np.linspace(2, 4, 20))))
        for i, dv in enumerate(ds.design_variables):
            self.assertIsInstance(dv, IntegerVariable)
            x[:, i] = ds._round_equally_distributed(x[:, i], dv.lower, dv.upper)

        x1, x1_counts = np.unique(x[:, 0], return_counts=True)
        self.assertTrue(np.all(x1 == [0, 1, 2, 3, 4, 5]))
        x1_counts = x1_counts / np.sum(x1_counts)
        self.assertTrue(np.all(np.abs(x1_counts - np.mean(x1_counts)) <= .05))

        x2, x2_counts = np.unique(x[:, 1], return_counts=True)
        self.assertTrue(np.all(x2 == [-1, 0, 1]))
        x2_counts = x2_counts / np.sum(x2_counts)
        self.assertTrue(np.all(np.abs(x2_counts - np.mean(x2_counts)) <= .05))

        x3, x3_counts = np.unique(x[:, 2], return_counts=True)
        self.assertTrue(np.all(x3 == [2, 3, 4]))
        x3_counts = x3_counts / np.sum(x3_counts)
        self.assertTrue(np.all(np.abs(x3_counts - np.mean(x3_counts)) <= .05))

    def test_base_design_space(self):
        ds = BaseDesignSpace([
            CategoricalVariable(['A', 'B']),
            IntegerVariable(0, 3),
            FloatVariable(-.5, .5),
        ])
        self.assertEqual(ds.get_x_types(), [(XType.ENUM, 2), XType.ORD, XType.FLOAT])
        self.assertEqual(ds.get_x_limits(), [['A', 'B'], (0, 3), (-.5, .5)])
        self.assertTrue(np.all(ds.get_num_bounds() == [[0, 1], [0, 3], [-.5, .5]]))
        self.assertTrue(np.all(ds.get_unfolded_num_bounds() == [[0, 1], [0, 1], [0, 3], [-.5, .5]]))

        x = np.array([
            [0, 0, 0],
            [1, 2, .5],
            [0, 3, .5],
        ])
        is_acting = np.array([
            [True, True, False],
            [True, False, True],
            [False, True, True],
        ])

        x_unfolded, is_acting_unfolded = ds.unfold_x(x, is_acting)
        self.assertTrue(np.all(x_unfolded == [
            [1, 0, 0, 0],
            [0, 1, 2, .5],
            [1, 0, 3, .5],
        ]))
        self.assertEqual(is_acting_unfolded.dtype, bool)
        self.assertTrue(np.all(is_acting_unfolded == [
            [True, True, True, False],
            [True, True, False, True],
            [False, False, True, True],
        ]))

        x_folded, is_acting_folded = ds.fold_x(x_unfolded, is_acting_unfolded)
        self.assertTrue(np.all(x_folded == x))
        self.assertTrue(np.all(is_acting_folded == is_acting))

    def test_legacy_design_space(self):
        ds = LegacyDesignSpace(
            x_limits=[[0, 1], (0, 2), ['05', '09', '23', '27'], ['C172', 'PA28', 'DA20', 'A210']],
            x_types=[XType.FLOAT, XType.ORD, XType.ORD, (XType.ENUM, 4)],
        )
        self.assertEqual(len(ds.design_variables), 4)
        self.assertTrue(np.all(~ds.is_conditionally_acting))

        dv0, dv1, dv2, dv3 = ds.design_variables
        self.assertIsInstance(dv0, FloatVariable)
        self.assertEqual(dv0.lower, 0)
        self.assertEqual(dv0.upper, 1)

        self.assertIsInstance(dv1, IntegerVariable)
        self.assertEqual(dv1.lower, 0)
        self.assertEqual(dv1.upper, 2)

        self.assertIsInstance(dv2, OrdinalVariable)
        self.assertEqual(dv2.values, ['05', '09', '23', '27'])

        self.assertIsInstance(dv3, CategoricalVariable)
        self.assertEqual(dv3.values, ['C172', 'PA28', 'DA20', 'A210'])

        ds.seed = 42
        x, is_acting = ds.sample_valid_x(3)
        self.assertEqual(x.shape, (3, 4))
        self.assertTrue(np.all(np.abs(x - np.array([
            [.125, 2, 2, 0],
            [.622, 0, 3, 1],
            [.727, 1, 0, 3],
        ])) < 1e-2))
        self.assertEqual(is_acting.shape, x.shape)
        self.assertTrue(np.all(is_acting))

        x_sampled_externally = LHS(xlimits=ds.get_unfolded_num_bounds(), criterion='ese', random_state=42)(3)
        x_corr, is_acting_corr = ds.correct_get_acting(x_sampled_externally)
        x_corr, is_acting_corr = ds.fold_x(x_corr, is_acting_corr)
        self.assertTrue(np.all(x_corr == x))
        self.assertTrue(np.all(is_acting_corr))

        self.assertTrue(str(ds))
        self.assertTrue(repr(ds))

        ds = LegacyDesignSpace(x_limits=np.array([[0, 1], [2, 3], [4, 5]]))
        self.assertEqual(len(ds.design_variables), 3)
        self.assertTrue(all(isinstance(dv, FloatVariable) for dv in ds.design_variables))
        self.assertEqual(ds.design_variables[0].get_limits(), (0, 1))
        self.assertEqual(ds.get_x_limits(), [(0, 1), (2, 3), (4, 5)])
        self.assertTrue(np.all(ds.get_num_bounds() == np.array([[0, 1], [2, 3], [4, 5]])))

    def test_design_space(self):
        ds = DesignSpace([
            CategoricalVariable(['A', 'B', 'C']),
            OrdinalVariable(['E', 'F']),
            IntegerVariable(-1, 2),
            FloatVariable(.5, 1.5),
        ], seed=42)
        self.assertEqual(len(ds.design_variables), 4)
        self.assertEqual(len(ds._cs.get_hyperparameters()), 4)
        self.assertTrue(np.all(~ds.is_conditionally_acting))

        x, is_acting = ds.sample_valid_x(3)
        self.assertEqual(x.shape, (3, 4))
        self.assertTrue(np.all(np.abs(x - np.array([
            [1, 0, 0, .834],
            [2, 0, -1, .6434],
            [2, 0, 0, 1.151],
        ])) < 1e-2))
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(is_acting.shape, x.shape)

        self.assertEqual(ds.decode_values(x, i_dv=0), ['B', 'C', 'C'])
        self.assertEqual(ds.decode_values(x, i_dv=1), ['E', 'E', 'E'])
        self.assertEqual(ds.decode_values(np.array([0, 1, 2]), i_dv=0), ['A', 'B', 'C'])
        self.assertEqual(ds.decode_values(np.array([0, 1]), i_dv=1), ['E', 'F'])

        x_corr, is_act_corr = ds.correct_get_acting(x)
        self.assertTrue(np.all(x_corr == x))
        self.assertTrue(np.all(is_act_corr == is_acting))

        x_sampled_externally = LHS(xlimits=ds.get_unfolded_num_bounds(), criterion='ese', random_state=42)(3)
        x_corr, is_acting_corr = ds.correct_get_acting(x_sampled_externally)
        x_corr, is_acting_corr = ds.fold_x(x_corr, is_acting_corr)
        self.assertTrue(np.all(np.abs(x_corr-np.array([
            [2, 0, -1, 1.342],
            [0, 1, 0, .552],
            [1, 1, 2, 1.157],
        ])) < 1e-3))
        self.assertTrue(np.all(is_acting_corr))

        x_unfolded, is_acting_unfolded = ds.sample_valid_x(3, unfolded=True)
        self.assertEqual(x_unfolded.shape, (3, 6))
        self.assertTrue(np.all(np.abs(x_unfolded - np.array([
            [1, 0, 0, 1, -1, .932],
            [0, 0, 1, 1, 0, .791],
            [0, 0, 1, 1, 1, 1.112],
        ])) < 1e-2))

        self.assertTrue(str(ds))
        self.assertTrue(repr(ds))

    def test_design_space_hierarchical(self):
        ds = DesignSpace([
            CategoricalVariable(['A', 'B', 'C']),  # x0
            CategoricalVariable(['E', 'F']),  # x1
            IntegerVariable(0, 1),  # x2
            FloatVariable(0, 1),  # x3
        ], seed=42)
        ds.declare_decreed_var(decreed_var=3, meta_var=0, meta_value='A')  # Activate x3 if x0 == A
        ds.add_value_constraint(var1=0, value1='C', var2=1, value2='F')  # Prevent a == C and b == F

        x_cartesian = np.array(list(itertools.product([0, 1, 2], [0, 1], [0, 1], [.25, .75])))
        self.assertEqual(x_cartesian.shape, (24, 4))

        self.assertTrue(np.all(ds.is_conditionally_acting == [False, False, False, True]))

        x, is_acting = ds.correct_get_acting(x_cartesian)
        _, is_unique = np.unique(x, axis=0, return_index=True)
        self.assertEqual(len(is_unique), 14)
        self.assertTrue(np.all(x[is_unique, :] == np.array([
            [0, 0, 0, .25],
            [0, 0, 0, .75],
            [0, 0, 1, .25],
            [0, 0, 1, .75],
            [0, 1, 0, .25],
            [0, 1, 0, .75],
            [0, 1, 1, .25],
            [0, 1, 1, .75],
            [1, 0, 0, .5],
            [1, 0, 1, .5],
            [1, 1, 0, .5],
            [1, 1, 1, .5],
            [2, 0, 0, .5],
            [2, 0, 1, .5],
        ])))
        self.assertTrue(np.all(is_acting[is_unique, :] == np.array([
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
        ])))

        x_sampled, is_acting_sampled = ds.sample_valid_x(100)
        assert x_sampled.shape == (100, 4)
        x_sampled[is_acting_sampled[:, 3], 3] = np.round(x_sampled[is_acting_sampled[:, 3], 3])

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

    def test_check_conditionally_acting(self):

        class WrongDesignSpace(DesignSpace):

            def _is_conditionally_acting(self) -> np.ndarray:
                return np.zeros((self.n_dv,), dtype=bool)

        ds = WrongDesignSpace([
            CategoricalVariable(['A', 'B', 'C']),  # x0
            CategoricalVariable(['E', 'F']),  # x1
            IntegerVariable(0, 1),  # x2
            FloatVariable(0, 1),  # x3
        ], seed=42)
        ds.declare_decreed_var(decreed_var=3, meta_var=0, meta_value='A')  # Activate x3 if x0 == A

        self.assertRaises(RuntimeError, lambda: ds.sample_valid_x(10))

    def test_restrictive_value_constraint(self):
        ds = DesignSpace([
            IntegerVariable(0, 2),
            IntegerVariable(0, 2),
        ])
        assert ds._cs.get_hyperparameters()[0].default_value == 1

        ds.add_value_constraint(var1=0, value1=1, var2=0, value2=1)
        ds.sample_valid_x(100)

        x_cartesian = np.array(list(itertools.product([0, 1, 2], [0, 1, 2])))
        ds.correct_get_acting(x_cartesian)


if __name__ == '__main__':
    unittest.main()
