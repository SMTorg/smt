from libcpp.vector cimport vector
from cpython cimport array
import numpy as np
cimport numpy as np


cdef extern from "rmtb.hpp":
  cdef cppclass RMTB:
    RMTB() except +
    void setup(int nx, double * lower, double * upper, int * order_list, int * ncp_list)
    void compute_ext_dist(int n, int nterm, double * x, double * dx)
    void compute_quadrature_points(int n, int * nelem_list, double * x)
    void compute_jac(int ix1, int ix2, int n, double * t, double * data, int * rows, int * cols)

cdef extern from "rmtc.hpp":
  cdef cppclass RMTC:
    RMTC() except +
    void setup(int nx, double * lower, double * upper, int * nelem_list, int * nterm_list)
    void compute_ext_dist(int n, int nterm, double * x, double * dx)
    void compute_quadrature_points(int n, int * nelem_list, double * x)
    void compute_coeff2nodal(double * mtx)
    void compute_uniq2elem(double * data, int * rows, int * cols)
    void compute_full_from_block(double * mtx, double * data, int * rows, int * cols)
    void compute_jac(int ix1, int ix2, int n, double * x, double * data, int * rows, int * cols)

cdef class PyRMTB:

    cdef RMTB *thisptr
    def __cinit__(self):
        self.thisptr = new RMTB()
    def __dealloc__(self):
        del self.thisptr
    def setup(self, int nx,
            np.ndarray[double] lower, np.ndarray[double] upper,
            np.ndarray[int] order_list, np.ndarray[int] ncp_list):
        self.thisptr.setup(nx, &lower[0], &upper[0], &order_list[0], &ncp_list[0])
    def compute_ext_dist(self, int n, int nterm, np.ndarray[double] x, np.ndarray[double] dx):
        self.thisptr.compute_ext_dist(n, nterm, &x[0], &dx[0])
    def compute_quadrature_points(self, int n, np.ndarray[int] nelem_list, np.ndarray[double] x):
        self.thisptr.compute_quadrature_points(n, &nelem_list[0], &x[0])
    def compute_jac(self, int ix1, int ix2, int n, np.ndarray[double] t,
            np.ndarray[double] data, np.ndarray[int] rows, np.ndarray[int] cols):
        self.thisptr.compute_jac(ix1, ix2, n, &t[0], &data[0], &rows[0], &cols[0])

cdef class PyRMTC:

    cdef RMTC *thisptr
    def __cinit__(self):
        self.thisptr = new RMTC()
    def __dealloc__(self):
        del self.thisptr
    def setup(self, int nx,
            np.ndarray[double] lower, np.ndarray[double] upper,
            np.ndarray[int] nelem_list, np.ndarray[int] nterm_list):
        self.thisptr.setup(nx, &lower[0], &upper[0], &nelem_list[0], &nterm_list[0])
    def compute_ext_dist(self, int n, int nterm, np.ndarray[double] x, np.ndarray[double] dx):
        self.thisptr.compute_ext_dist(n, nterm, &x[0], &dx[0])
    def compute_quadrature_points(self, int n, np.ndarray[int] nelem_list, np.ndarray[double] x):
        self.thisptr.compute_quadrature_points(n, &nelem_list[0], &x[0])
    def compute_coeff2nodal(self, np.ndarray[double] mtx):
        self.thisptr.compute_coeff2nodal(&mtx[0])
    def compute_uniq2elem(self,
            np.ndarray[double] data, np.ndarray[int] rows, np.ndarray[int] cols):
        self.thisptr.compute_uniq2elem(&data[0], &rows[0], &cols[0])
    def compute_full_from_block(self, np.ndarray[double] mtx,
            np.ndarray[double] data, np.ndarray[int] rows, np.ndarray[int] cols):
        self.thisptr.compute_full_from_block(&mtx[0], &data[0], &rows[0], &cols[0])
    def compute_jac(self, int ix1, int ix2, int n, np.ndarray[double] x,
            np.ndarray[double] data, np.ndarray[int] rows, np.ndarray[int] cols):
        self.thisptr.compute_jac(ix1, ix2, n, &x[0], &data[0], &rows[0], &cols[0])
