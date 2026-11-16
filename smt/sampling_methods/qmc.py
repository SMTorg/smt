"""
Author: Paul Saves <paul.saves@ird.fr>

This package is distributed under New BSD license.

Quasi-Monte Carlo low-discrepancy sampling methods (Sobol, Halton).
"""

import inspect
from numbers import Integral

import numpy as np
from scipy.stats import qmc

from smt.sampling_methods.sampling_method import ScaledSamplingMethod


def _build_qmc_sampler(sampler_cls, nx, scramble, rng):
    """Instantiate a SciPy QMC sampler across SciPy version signatures."""
    params = inspect.signature(sampler_cls).parameters
    kwargs = {"d": nx, "scramble": scramble}
    if "rng" in params:
        kwargs["rng"] = rng
    elif "seed" in params:
        kwargs["seed"] = rng
    return sampler_cls(**kwargs)


class _QMCSampling(ScaledSamplingMethod):
    """Shared base class for SciPy Quasi-Monte Carlo sampling methods."""

    _sampler_cls = None

    def _initialize(self, **kwargs):
        self.options.declare(
            "scramble",
            True,
            types=bool,
            desc="Whether to apply scrambling to the QMC sequence",
        )
        self.options.declare(
            "seed",
            types=(type(None), Integral, np.random.Generator),
            desc="Numpy Generator or integer seed controlling sequence scrambling",
        )
        self.options.update(kwargs)

        self._sampler = None
        self._sampler_cfg = None

    def _get_sampler(self, nx):
        seed = self.options["seed"]
        seed_key = seed if isinstance(seed, Integral) else id(seed)
        cfg = (nx, bool(self.options["scramble"]), seed_key)
        if self._sampler is None or self._sampler_cfg != cfg:
            rng = (
                np.random.default_rng(seed)
                if seed is not None
                else np.random.default_rng()
            )
            self._sampler = _build_qmc_sampler(
                self._sampler_cls,
                nx=nx,
                scramble=bool(self.options["scramble"]),
                rng=rng,
            )
            self._sampler_cfg = cfg
        return self._sampler

    def _compute(self, nt):
        """
        Compute quasi-random samples in the unit hypercube ``[0, 1]^nx``.

        Parameters
        ----------
        nt : int
            Number of points requested.

        Returns
        -------
        ndarray[nt, nx]
            Sampling locations in ``[0, 1]^nx``.
        """
        nx = self.options["xlimits"].shape[0]
        if nt == 0:
            return np.empty((0, nx), dtype=float)
        return self._get_sampler(nx).random(int(nt))


class Sobol(_QMCSampling):
    """Sobol low-discrepancy sequence sampling."""

    _sampler_cls = qmc.Sobol


class Halton(_QMCSampling):
    """Halton low-discrepancy sequence sampling."""

    _sampler_cls = qmc.Halton
