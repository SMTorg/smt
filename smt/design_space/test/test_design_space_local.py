import unittest
import os

os.environ["FORCE_RUN_LOCAL"] = "1"


class TestDSLocal(unittest.TestCase):
    def test_design_variables(self):
        os.environ["FORCE_RUN_LOCAL"] = "1"
        # Load all tests from test_design_space.py
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=".", pattern="test_design_space.py")
        # Run the tests
        runner = unittest.TextTestRunner()
        runner.run(suite)


if __name__ == "__main__":
    unittest.main()
