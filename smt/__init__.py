import glob
from os.path import join, dirname

__version__ = "2.7.0"

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
repo_root = os.path.abspath(os.path.dirname(__file__))
