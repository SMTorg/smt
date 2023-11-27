import os
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

    def test_random_state(self):
        xlimits = np.array([[0.0, 4.0], [0.0, 3.0]])
        num = 10
        sampling = LHS(xlimits=xlimits, criterion="ese", random_state=42)
        doe1 = sampling(num)
        doe2 = sampling(num)

        # Should not generate the same doe
        self.assertFalse(np.allclose(doe1, doe2))

        # Another LHS with same initialization should generate the same sequence of does
        sampling = LHS(
            xlimits=xlimits, criterion="ese", random_state=np.random.RandomState(42)
        )
        doe3 = sampling(num)
        doe4 = sampling(num)
        self.assertTrue(np.allclose(doe1, doe3))
        self.assertTrue(np.allclose(doe2, doe4))

    @unittest.skipIf(int(os.getenv("RUN_SLOW", 0)) < 1, "too slow")
    def test_expand_lhs(self):
        import numpy as np

        num = 100
        new_list = np.linspace(1, 5, 5) * num

        for i in range(len(new_list)):
            xlimits = np.array([[0.0, 4.0], [0.0, 3.0], [0.0, 3.0], [1.0, 5.0]])
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


if __name__ == "__main__":
    unittest.main()
