import unittest

import matplotlib

matplotlib.use("Agg")

try:
    from smt.surrogate_models import RMTB, RMTC

    compiled_available = True
except:
    compiled_available = False


class Test(unittest.TestCase):
    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtb(self):
        from smt.examples.one_D_step import run_one_D_step_rmtb

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        from smt.examples.one_D_step import run_one_D_step_rmtc
