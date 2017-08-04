from libcpp.vector cimport vector
from cpython cimport array
import numpy as np
cimport numpy as np


cdef extern from "rmts.hpp":
  cdef cppclass RMTS:
    RMTS() except +
    void setup(int nx, double * lower, double * upper)
    void compute_ext_dist(int n, int nterm, double * x, double * dx)
    # void compute_jac(int n, double * x, double * jac)
    # void compute_jac_derivs(int n, int kx, double* x, double* jac)


cdef class PyRMTS:

    cdef RMTS *thisptr
    def __cinit__(self):
        self.thisptr = new RMTS()
    def __dealloc__(self):
        del self.thisptr
    def setup(self, int nx, np.ndarray[double] lower, np.ndarray[double] upper):
        self.thisptr.setup(nx, &lower[0], &upper[0])
    def compute_ext_dist(self, int n, int nterm, np.ndarray[double] x, np.ndarray[double] dx):
        self.thisptr.compute_ext_dist(n, nterm, &x[0], &dx[0])
    # def compute_jac_derivs(self, int n, int kx, np.ndarray[double] x, np.ndarray[double] jac):
    #     self.thisptr.compute_jac_derivs(n, kx, &x[0], &jac[0])
