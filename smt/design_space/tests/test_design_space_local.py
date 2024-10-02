import unittest
import os

os.environ["RUN_PLAIN_DESIGN_SPACE_TEST"] = "1"
from test_design_space import Test  # Import the existing test class


class TestDSLocal(Test):
    def test_design_variables(self):
        os.environ["RUN_PLAIN_DESIGN_SPACE_TEST"] = "1"
        # Load all tests from test_design_space.py
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir=".", pattern="test_design_space.py")
        # Run the tests
        runner = unittest.TextTestRunner()
        runner.run(suite)
        del os.environ["RUN_PLAIN_DESIGN_SPACE_TEST"]


if __name__ == "__main__":
    unittest.main()
