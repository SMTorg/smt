from libcpp.vector cimport vector
from cpython cimport array
import numpy as np
cimport numpy as np


cdef extern from "rbf.hpp":
  cdef cppclass RBF:
    RBF() except +
    void setup(int nx, int nt, int num_dof, int poly_degree, double * d0, double * xt)
    void compute_jac(int n, double* x, double* jac)
    void compute_jac_derivs(int n, int kx, double* x, double* jac)


cdef class PyRBF:

    cdef RBF *thisptr
    def __cinit__(self):
        self.thisptr = new RBF()
    def __dealloc__(self):
        del self.thisptr
    def setup(self,
            int nx, int nt, int num_dof, int poly_degree,
            np.ndarray[double] d0, np.ndarray[double] xt):
        self.thisptr.setup(nx, nt, num_dof, poly_degree, &d0[0], &xt[0])
    def compute_jac(self, int n, np.ndarray[double] x, np.ndarray[double] jac):
        self.thisptr.compute_jac(n, &x[0], &jac[0])
    def compute_jac_derivs(self, int n, int kx, np.ndarray[double] x, np.ndarray[double] jac):
        self.thisptr.compute_jac_derivs(n, kx, &x[0], &jac[0])
