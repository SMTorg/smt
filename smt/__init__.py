from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("smt")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "surrogate_models",
    "kernels",
    "design_space",
    "applications",
    "examples",
    "sampling_methods",
    "utils",
    "tests",
    "src",
    "problems",
]
