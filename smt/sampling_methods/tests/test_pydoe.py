import unittest
import numpy as np

from smt.sampling_methods import pydoe


class TestPyDOE3(unittest.TestCase):
    def test_bbdesign(self):
        xlimits = np.array([[2.0, 5], [-5, 1], [0, 3]])
        sampling = pydoe.BoxBehnken(xlimits=xlimits)
        actual = sampling()
        self.assertEqual((15, 3), actual.shape)
        print(actual)
        expected = [
            [2.0, -5.0, 1.5],
            [5.0, -5.0, 1.5],
            [2.0, 1.0, 1.5],
            [5.0, 1.0, 1.5],
            [2.0, -2.0, 0.0],
            [5.0, -2.0, 0.0],
            [2.0, -2.0, 3.0],
            [5.0, -2.0, 3.0],
            [3.5, -5.0, 0.0],
            [3.5, 1.0, 0.0],
            [3.5, -5.0, 3.0],
            [3.5, 1.0, 3.0],
            [3.5, -2.0, 1.5],
            [3.5, -2.0, 1.5],
            [3.5, -2.0, 1.5],
        ]

        np.testing.assert_allclose(actual, expected)

    def test_gsd1(self):
        xlimits = np.array([[5, 11], [0, 6], [-3, 4]])
        sampling = pydoe.Gsd(xlimits=xlimits, levels=[3, 4, 6], reduction=4)
        actual = sampling()
        self.assertEqual((18, 3), actual.shape)
        print(actual)
        expected = [
            [5, 0, -3],
            [5, 0, 2.6],
            [5, 2, -1.6],
            [5, 2, 4],
            [5, 4, -0.2],
            [5, 6, 1.2],
            [8, 0, -1.6],
            [8, 0, 4],
            [8, 2, -0.2],
            [8, 4, 1.2],
            [8, 6, -3],
            [8, 6, 2.6],
            [11, 0, -0.2],
            [11, 2, 1.2],
            [11, 4, -3],
            [11, 4, 2.6],
            [11, 6, -1.6],
            [11, 6, 4],
        ]

        np.testing.assert_allclose(actual, expected)

    def test_factorial(self):
        xlimits = np.array([[4, 10], [-3, 3], [5, 7]])
        sampling = pydoe.Factorial(xlimits=xlimits, levels=[2, 4, 3])
        actual = sampling()
        self.assertEqual((24, 3), actual.shape)
        print(actual)
        expected = [
            [4.0, -3.0, 5.0],
            [10.0, -3.0, 5.0],
            [4.0, -1.0, 5.0],
            [10.0, -1.0, 5.0],
            [4.0, 1.0, 5.0],
            [10.0, 1.0, 5.0],
            [4.0, 3.0, 5.0],
            [10.0, 3.0, 5.0],
            [4.0, -3.0, 6.0],
            [10.0, -3.0, 6.0],
            [4.0, -1.0, 6.0],
            [10.0, -1.0, 6.0],
            [4.0, 1.0, 6.0],
            [10.0, 1.0, 6.0],
            [4.0, 3.0, 6.0],
            [10.0, 3.0, 6.0],
            [4.0, -3.0, 7.0],
            [10.0, -3.0, 7.0],
            [4.0, -1.0, 7.0],
            [10.0, -1.0, 7.0],
            [4.0, 1.0, 7.0],
            [10.0, 1.0, 7.0],
            [4.0, 3.0, 7.0],
            [10.0, 3.0, 7.0],
        ]
        np.testing.assert_allclose(actual, expected)

    def test_plackett_burman(self):
        xlimits = np.array([[2, 5], [-5, 1], [0, 3], [4, 8], [-1, 2]])
        sampling = pydoe.PlackettBurman(xlimits=xlimits)
        actual = sampling()
        self.assertEqual((8, 5), actual.shape)
        print(actual)
        expected = [
            [2.0, -5.0, 3.0, 4.0, 2.0],
            [5.0, -5.0, 0.0, 4.0, -1.0],
            [2.0, 1.0, 0.0, 4.0, 2.0],
            [5.0, 1.0, 3.0, 4.0, -1.0],
            [2.0, -5.0, 3.0, 8.0, -1.0],
            [5.0, -5.0, 0.0, 8.0, 2.0],
            [2.0, 1.0, 0.0, 8.0, -1.0],
            [5.0, 1.0, 3.0, 8.0, 2.0],
        ]

        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
