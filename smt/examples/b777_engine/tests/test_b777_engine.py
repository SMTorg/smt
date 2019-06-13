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
        from smt.examples.b777_engine import run_b777_engine_rmtb

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        from smt.examples.b777_engine import run_b777_engine_rmtc


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Test().test_rmtc()
    plt.savefig("test.pdf")
