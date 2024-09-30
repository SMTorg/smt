import unittest
import os

os.environ["FORCE_RUN_PLAIN_BASICS"] = "1"
from test_design_space import Test  # Import the existing test class


class TestDSLocal(Test):
    def test_design_variables(self):
        os.environ["FORCE_RUN_PLAIN_BASICS"] = "1"
        # Load all tests from test_design_space.py
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=".", pattern="test_design_space.py")
        # Run the tests
        runner = unittest.TextTestRunner()
        runner.run(suite)
        del os.environ["FORCE_RUN_PLAIN_BASICS"]


if __name__ == "__main__":
    unittest.main()
