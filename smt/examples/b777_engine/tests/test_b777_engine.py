import unittest

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

try:
    from smt.surrogate_models import RMTB, RMTC

    NO_COMPILED = True
except ImportError:
    NO_COMPILED = False


class Test(unittest.TestCase):
    @unittest.skipIf(
        NO_COMPILED or NO_MATPLOTLIB,
        "C compilation failed or matplotlib not installed",
    )
    def test_rmtb(self):
        from smt.examples.b777_engine import run_b777_engine_rmtb

    @unittest.skipIf(
        NO_COMPILED or NO_MATPLOTLIB,
        "C compilation failed or matplotlib not installed",
    )
    def test_rmtc(self):
        from smt.examples.b777_engine import run_b777_engine_rmtc


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Test().test_rmtc()
    plt.savefig("test.pdf")
