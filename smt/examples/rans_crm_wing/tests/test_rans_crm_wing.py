import unittest

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True

try:
    from smt.surrogate_models import RMTB, RMTC  # noqa: F401

    NO_COMPILED = True
except ImportError:
    NO_COMPILED = False


class Test(unittest.TestCase):
    @unittest.skipIf(
        NO_COMPILED or NO_MATPLOTLIB,
        "C compilation failed or matplotlib not installed",
    )
    def test_rmtb(self):
        # just check import
        from smt.examples.rans_crm_wing import run_rans_crm_wing_rmtb  # noqa: F401

    @unittest.skipIf(
        NO_COMPILED or NO_MATPLOTLIB,
        "C compilation failedor matplotlib not installed",
    )
    def test_rmtc(self):
        # just check import
        from smt.examples.rans_crm_wing import run_rans_crm_wing_rmtc  # noqa: F401


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Test().test_rmtb()
    plt.savefig("test.pdf")
