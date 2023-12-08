Developer API for sampling methods
==================================

SamplingMethod
--------------

A base class for all sampling methods in SMT.

.. autoclass:: smt.sampling_methods.sampling_method.SamplingMethod

  .. automethod:: smt.sampling_methods.sampling_method.SamplingMethod._initialize

  .. automethod:: smt.sampling_methods.sampling_method.SamplingMethod._compute

ScaledSamplingMethod
--------------------

Conveniently, if a sampling method generates samples in the [0, 1] hypercube,
one can inherit from the subclass `ScaledSamplingMethod` which 
automates the scaling from unit hypercube to the input space (i.e. xlimits).

.. autoclass:: smt.sampling_methods.sampling_method.ScaledSamplingMethod

  .. automethod:: smt.sampling_methods.sampling_method.ScaledSamplingMethod._compute
