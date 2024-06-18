Gausian process regression
==========================

SMT implements several surrogate models related to Gaussian process regression:

* `KRG` implements classic gaussian process regression. 
* `KPLS` and `KPLSK` are variants using PLS dimension reduction to address high-dimensional training data.
* `GPX` is a re-implementation of `KRG` and `KPLS` using Rust for faster training/prediction operations.
* `GEKPLS` leverages on derivatives training data to improve the surrogate model quality.
* `MGP` takes into account the uncertainty of the hyperparameters defined as a density function.   
* `SGP` implements sparse methods allowing to deal with large training dataset as others implementations have a time complexity of :math:`O(n^3)` as well as a :math:`O(n^2)` memory cost in the number :math:`n` of training points.

Here below, the links to the dedicated pages:  

.. toctree::
   :maxdepth: 1
   :titlesonly:

   gpr/krg
   gpr/kpls
   gpr/kplsk
   gpr/gpx
   gpr/gekpls
   gpr/mgp
   gpr/sgp
