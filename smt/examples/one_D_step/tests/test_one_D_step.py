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
        from smt.examples.one_D_step import run_one_D_step_rmtb

    @unittest.skipIf(
        NO_COMPILED or NO_MATPLOTLIB,
        "C compilation failed or matplotlib not installed",
    )
    def test_rmtc(self):
        from smt.examples.one_D_step import run_one_D_step_rmtc
