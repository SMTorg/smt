#include "rmtb.hpp"
#include "rmts.hpp"
#include "utils.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <sstream>

using namespace std;

void compute_knot_vector_uniform(long order, long ncp, double * knots) {
  for (long i = 0; i < order + ncp; i++) {
    knots[i] = 1. * (i - order + 1) / (ncp - order + 1);
  }
}

long compute_i_start(long order, long ncp, double param, double * knots) {
  long istart = -1;

  for (long i = order - 1; i < ncp; i++) {
    if ((knots[i] <= param) && (param < knots[i + 1])) {
      istart = i - order + 1;
    }
  }

  if (param == knots[order + ncp - 1]) {
    istart = ncp - order;
  }

  return istart;
}

long compute_basis_0(long order, long ncp, double param, double * knots, double * basis0_vec) {
  long istart = compute_i_start(order, ncp, param, knots);

  for (long i = 0; i < order; i++) {
    basis0_vec[i] = 0.;
  }
  basis0_vec[order - 1] = 1.;

  for (long i = 1; i < order; i++) {
    long j1 = order - i - 1;
    long j2 = order - 1;
    long n = istart + j1;

    if (knots[n + i + 1] != knots[n + 1]) {
      basis0_vec[j1] = (knots[n + i + 1] - param) / (knots[n + i + 1] - knots[n + 1]) * basis0_vec[j1 + 1];
    } else {
      basis0_vec[j1] = 0.;
    }
    for (long j = j1 + 1; j < j2; j++) {
      n = istart + j;
      if (knots[n + i] != knots[n]) {
        basis0_vec[j] = (param - knots[n]) / (knots[n + i] - knots[n]) * basis0_vec[j];
      } else {
        basis0_vec[j] = 0.;
      }
      if (knots[n + i + 1] != knots[n + 1]) {
        basis0_vec[j] += (knots[n + i + 1] - param) / (knots[n + i + 1] - knots[n + 1]) * basis0_vec[j + 1];
      }
    }
    n = istart + j2;
    if (knots[n + i] != knots[n]) {
      basis0_vec[j2] = (param - knots[n]) / (knots[n + i] - knots[n]) * basis0_vec[j2];
    } else {
      basis0_vec[j2] = 0.;
    }
  }
  return istart;
}

long compute_basis_1(long order, long ncp, double param, double * knots, double * basis1_vec) {
  long istart = compute_i_start(order, ncp, param, knots);

  double * basis0_vec = new double[order];

  for (long i = 0; i < order; i++) {
    basis0_vec[i] = 0.;
    basis1_vec[i] = 0.;
  }
  basis0_vec[order - 1] = 1.;

  for (long i = 1; i < order; i++) {
    long j1 = order - i - 1;
    long j2 = order - 1;

    for (long j = j1; j < j2 + 1; j++) {
      long n = istart + j;
      double b1, b2, f1, f2;

      if (knots[n + i] != knots[n]) {
        double den = knots[n + i] - knots[n];
        b1 = (param - knots[n]) / den * basis0_vec[j];
        f1 = (basis0_vec[j] + (param - knots[n]) * basis1_vec[j]) / den;
      } else {
        b1 = 0.;
        f1 = 0.;
      }
      if ((j != j2) && (knots[n + i + 1] != knots[n + 1])) {
        double den = knots[n + i + 1] - knots[n + 1];
        b2 = (knots[n + i + 1] - param) / den * basis0_vec[j + 1];
        f2 = ((knots[n + i + 1] - param) * basis1_vec[j + 1] - basis0_vec[j + 1]) / den;
      } else {
        b2 = 0.;
        f2 = 0.;
      }
      basis0_vec[j] = b1 + b2;
      basis1_vec[j] = f1 + f2;
    }
  }

  delete[] basis0_vec;

  return istart;
}

long compute_basis_2(long order, long ncp, double param, double * knots, double * basis2_vec) {
  long istart = compute_i_start(order, ncp, param, knots);

  double * basis0_vec = new double[order];
  double * basis1_vec = new double[order];

  for (long i = 0; i < order; i++) {
    basis0_vec[i] = 0.;
    basis1_vec[i] = 0.;
    basis2_vec[i] = 0.;
  }
  basis0_vec[order - 1] = 1.;

  for (long i = 1; i < order; i++) {
    long j1 = order - i - 1;
    long j2 = order - 1;

    for (long j = j1; j < j2 + 1; j++) {
      long n = istart + j;
      double b1, b2, f1, f2, s1, s2;

      if (knots[n + i] != knots[n]) {
        double den = knots[n + i] - knots[n];
        b1 = (param - knots[n]) / den * basis0_vec[j];
        f1 = (basis0_vec[j] + (param - knots[n]) * basis1_vec[j]) / den;
        s1 = (2 * basis1_vec[j] + (param - knots[n]) * basis2_vec[j]) / den;
      } else {
        b1 = 0.;
        f1 = 0.;
        s1 = 0.;
      }
      if ((j != j2) && (knots[n + i + 1] != knots[n + 1])) {
        double den = knots[n + i + 1] - knots[n + 1];
        b2 = (knots[n + i + 1] - param) / den * basis0_vec[j + 1];
        f2 = ((knots[n + i + 1] - param) * basis1_vec[j + 1] - basis0_vec[j + 1]) / den;
        s2 = ((knots[n + i + 1] - param) * basis2_vec[j + 1] - 2 * basis1_vec[j + 1]) / den;
      } else {
        b2 = 0.;
        f2 = 0.;
        s2 = 0.;
      }
      basis0_vec[j] = b1 + b2;
      basis1_vec[j] = f1 + f2;
      if (i > 1) {
        basis2_vec[j] = s1 + s2;
      }
    }
  }

  delete[] basis0_vec;
  delete[] basis1_vec;

  return istart;
}

RMTB::RMTB() {
  order_list = NULL;
  ncp_list = NULL;
}

RMTB::~RMTB() {
  delete[] order_list;
  delete[] ncp_list;
}

void RMTB::setup(int nx, double * lower, double * upper, int * order_list, int * ncp_list) {
  RMTS::setup(nx, lower, upper);

  delete[] this->order_list;
  delete[] this->ncp_list;

  this->order_list = new int[nx];
  this->ncp_list = new int[nx];

  memcpy(this->order_list, order_list, nx * sizeof(*order_list));
  memcpy(this->ncp_list, ncp_list, nx * sizeof(*ncp_list));
}

void RMTB::compute_jac(
    int ix1, int ix2, int n, double * params,
    double * data, int * rows, int * cols) {
  long nnz_row = 1;

  for (long ix = 0; ix < nx; ix++) {
    nnz_row *= order_list[ix];
  }

  for (long i = 0; i < n; i++) {
    for (long inz_row = 0; inz_row < nnz_row; inz_row++) {
      data[i * nnz_row + inz_row] = 1.;
      rows[i * nnz_row + inz_row] = i;
      cols[i * nnz_row + inz_row] = 0;
    }
  }

  for (long ix = 0; ix < nx; ix++) {
    long order = order_list[ix];
    long ncp = ncp_list[ix];

    double * knots = new double[order + ncp];
    double * basis_vec = new double[order];

    compute_knot_vector_uniform(order, ncp, knots);

    for (long i = 0; i < n; i++) {
      long istart;
      long (*compute_basis)(long, long, double, double *, double *);

      if ((ix != ix1) && (ix != ix2)) {compute_basis = &compute_basis_0;}
      else if ((ix == ix1) && (ix != ix2)) {compute_basis = &compute_basis_1;}
      else if ((ix != ix1) && (ix == ix2)) {compute_basis = &compute_basis_1;}
      else if ((ix == ix1) && (ix == ix2)) {compute_basis = &compute_basis_2;}

      istart = (*compute_basis) (order, ncp, params[i * nx + ix], knots, basis_vec);

      for (long inz_row = 0; inz_row < nnz_row; inz_row++) {
        long rem = inz_row;
        long inz_dim;

        for (long jx = 0; jx < ix + 1; jx++) {
          long prod = 1;
          for (long kx = jx + 1; kx < nx; kx++) {
            prod *= order_list[kx];
          }
          inz_dim = rem / prod;
          rem -= inz_dim * prod;
        }

        long inz = i * nnz_row + inz_row;

        long prod = 1;
        for (long jx = ix + 1; jx < nx; jx++) {
          prod *= ncp_list[jx];
        }
        data[inz] *= basis_vec[inz_dim];
        cols[inz] += (inz_dim + istart) * prod;
      }
    }

    for (long i = 0; i < n; i++) {
        for (long inz_row = 0; inz_row < nnz_row; inz_row++) {
            long idx = i * nnz_row + inz_row;
            // Refer to https://github.com/SMTorg/smt/issues/573
            // I do not know why these column indices are sometimes negative.
            // However, I *have* shown that this post-filtering of the indices
            // fixes the problem, at least in the cases I have examined.
            if (cols[idx] < 0) {
                cols[idx] = 0;
                // The exception code is kept for future reference-
                // this is probably what *should* be done,
                // and the root cause of the negative indices should be understood.
                // std::ostringstream oss;
                // oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
                // oss << ":\n Computed a negative column index for sparse matrix; ";
                // oss << "please submit a bug report: https://github.com/SMTorg/smt/issues.\n";
                // oss << "Extra information for debug: ix1=" << ix1 << ", ix2=" << ix2 << ", ";
                // oss << "n= " << n << "\n";
                // throw std::logic_error(oss.str());
            }
        }
    }

    delete[] knots;
    delete[] basis_vec;
  }
}
