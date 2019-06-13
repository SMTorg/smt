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
        from smt.examples.rans_crm_wing import run_rans_crm_wing_rmtb

    @unittest.skipIf(not compiled_available, "C compilation failed")
    def test_rmtc(self):
        from smt.examples.rans_crm_wing import run_rans_crm_wing_rmtc


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Test().test_rmtb()
    plt.savefig("test.pdf")
