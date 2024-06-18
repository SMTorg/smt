Gausian process regression
==========================

Here surrogate models related to Gaussian process regressioh with the following high-level feature:

* `KRG` implements classic gaussian process regression. 
* `KPLS` and `KPLSK` are variants using PLS dimension reduction to address high-dimensional training data.
* `GPX` is a re-implementation of `KRG` and `KPLS` using Rust for faster training/prediction operations.
* `GEKPLS` leverage derivatives training data to improve the surrogate model quality.
* `SGP` implements sparse methods allowing to deal with large dataset as others implementations have a time complexity of :math:`O(n^3)` as well as a :math:`O(n^2)` memory cost where :math:`n` is the number of training points.
* `MGP` takes into account the uncertainty of the hyperparameters defined as a density function.   

.. toctree::
   :maxdepth: 1
   :titlesonly:

   gpr/krg
   gpr/kpls
   gpr/kplsk
   gpr/gpx
   gpr/gekpls
   gpr/sgp
   gpr/mgp
