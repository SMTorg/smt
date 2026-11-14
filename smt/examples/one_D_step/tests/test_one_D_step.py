import unittest

try:
    import matplotlib

    matplotlib.use("Agg")
    NO_MATPLOTLIB = False
except ImportError:
    NO_MATPLOTLIB = True


class Test(unittest.TestCase):
    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_rmtb(self):
        # just check import
        from smt.examples.one_D_step import run_one_D_step_rmtb  # noqa: F401

    @unittest.skipIf(NO_MATPLOTLIB, "Matplotlib not installed")
    def test_rmtc(self):
        # just check import
        from smt.examples.one_D_step import run_one_D_step_rmtc  # noqa: F401
