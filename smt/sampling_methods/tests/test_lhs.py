import unittest

import numpy as np

from smt.sampling_methods import LHS


class Test(unittest.TestCase):
    def test_lhs_ese(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        sampling = LHS(xlimits=xlimits, criterion="ese")
        num = 50
        x = sampling(num)

        self.assertEqual((50, 2), x.shape)

    def test_random_generator(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        num = 10
        sampling = LHS(xlimits=xlimits, criterion="ese", seed=42)
        doe1 = sampling(num)
        sampling = LHS(xlimits=xlimits, criterion="ese", seed=np.random.default_rng(42))
        doe2 = sampling(num)
        self.assertTrue(np.allclose(doe1, doe2))

    def test_deprecated_random_state(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        num = 10
        with self.assertWarns(DeprecationWarning):
            sampling = LHS(xlimits=xlimits, criterion="ese", random_state=42)
            _doe = sampling(num)
        with self.assertRaises(ValueError):
            sampling = LHS(
                xlimits=xlimits, criterion="ese", random_state=np.random.RandomState(42)
            )
            _doe = sampling(num)

    def test_random_state(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        num = 10
        sampling = LHS(xlimits=xlimits, criterion="ese", seed=42)
        doe1 = sampling(num)
        doe2 = sampling(num)

        # Should not generate the same doe
        self.assertFalse(np.allclose(doe1, doe2))

        # Another LHS with same initialization should generate the same sequence of does
        sampling = LHS(xlimits=xlimits, criterion="ese", seed=42)
        doe3 = sampling(num)
        doe4 = sampling(num)
        self.assertTrue(np.allclose(doe1, doe3))
        self.assertTrue(np.allclose(doe2, doe4))

    def test_expand_lhs(self):
        import numpy as np

        num = 10
        new_list = np.linspace(1, 5, 3) * num

        for i in range(len(new_list)):
            xlimits = np.array([[0.0, 4.0], [0.0, 3.0], [1.0, 5.0]])
            sampling = LHS(xlimits=xlimits, criterion="ese")

            x = sampling(num)
            new = int(new_list[i])
            new_num = num + new

            # We check the functionality with the "ese" optimization
            x_new = sampling.expand_lhs(x, new, method="ese")

            intervals = []
            subspace_bool = []
            for i in range(len(xlimits)):
                intervals.append(np.linspace(xlimits[i][0], xlimits[i][1], new_num + 1))

                subspace_bool.append(
                    [
                        [
                            intervals[i][j] < x_new[kk][i] < intervals[i][j + 1]
                            for kk in range(len(x_new))
                        ]
                        for j in range(len(intervals[i]) - 1)
                    ]
                )

                self.assertEqual(
                    True,
                    all(
                        [
                            subspace_bool[i][k].count(True) == 1
                            for k in range(len(subspace_bool[i]))
                        ]
                    ),
                )

    def test_expand_lhs_reproducibility(self):
        num = 5
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        sampling = LHS(xlimits=xlimits, criterion="ese", seed=42)

        x = sampling(num)
        new = 10

        x_new0 = sampling.expand_lhs(x, new)
        print(x_new0)

        # Test seeded expand against no seed expand
        x_new1 = sampling.expand_lhs(x, new, method="ese", seed=41)
        print(x_new1)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, x_new0, x_new1
        )

        # Test seeded expand against another differently seeded expand
        x_new2 = sampling.expand_lhs(x, new, method="ese", seed=42)
        print(x_new2)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, x_new1, x_new2
        )

        # Test expand reproducibility with same seed
        x_new3 = sampling.expand_lhs(x, new, method="ese", seed=42)
        print(x_new3)
        np.testing.assert_array_equal(x_new2, x_new3)

        # Test seeded expand with initial seed
        x_new4 = sampling.expand_lhs(x, new, method="ese", seed=41)
        print(x_new4)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, x_new3, x_new4
        )
        np.testing.assert_array_equal(x_new1, x_new4)

        # Test basic expand
        x_new5 = sampling.expand_lhs(x, new, seed=41)
        print(x_new5)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, x_new4, x_new5
        )

        # Test basic expand reproducibility
        x_new6 = sampling.expand_lhs(x, new, seed=41)
        print(x_new6)
        np.testing.assert_array_equal(x_new5, x_new6)


if __name__ == "__main__":
    unittest.main()
