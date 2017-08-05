#include "rmtc.hpp"
#include "rmts.hpp"
#include "utils.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cstring>

using namespace std;

RMTC::RMTC() {
  elem_list = NULL;
  term_list = NULL;
}

RMTC::~RMTC() {
  delete[] elem_list;
  delete[] term_list;
}

void RMTC::setup(int nx, double * lower, double * upper, int * elem_list, int * term_list) {
  RMTS::setup(nx, lower, upper);

  this->elem_list = new int[nx];
  this->term_list = new int[nx];

  memcpy(this->elem_list, elem_list, nx * sizeof(*elem_list));
  memcpy(this->term_list, term_list, nx * sizeof(*term_list));
}

// void RMTB::compute_jac(
//     int ix1, int ix2, int n, double * params,
//     double * data, int * rows, int * cols) {
//   int nnz_row = 1;
//
//   for (int ix = 0; ix < nx; ix++) {
//     nnz_row *= order_list[ix];
//   }
//
//   for (int i = 0; i < n; i++) {
//     for (int inz_row = 0; inz_row < nnz_row; inz_row++) {
//       data[i * nnz_row + inz_row] = 1.;
//       rows[i * nnz_row + inz_row] = i;
//       cols[i * nnz_row + inz_row] = 0;
//     }
//   }
//
//   for (int ix = 0; ix < nx; ix++) {
//     int order = order_list[ix];
//     int ncp = ncp_list[ix];
//
//     double knots[order + ncp];
//     double basis_vec[order];
//
//     compute_knot_vector_uniform(order, ncp, knots);
//
//     for (int i = 0; i < n; i++) {
//       int istart;
//       int (*compute_basis)(int, int, double, double *, double *);
//
//       if ((ix != ix1) && (ix != ix2)) {compute_basis = &compute_basis_0;}
//       else if ((ix == ix1) && (ix != ix2)) {compute_basis = &compute_basis_1;}
//       else if ((ix != ix1) && (ix == ix2)) {compute_basis = &compute_basis_1;}
//       else if ((ix == ix1) && (ix == ix2)) {compute_basis = &compute_basis_2;}
//
//       istart = (*compute_basis) (order, ncp, params[i * nx + ix], knots, basis_vec);
//
//       for (int inz_row = 0; inz_row < nnz_row; inz_row++) {
//         int rem = inz_row;
//         int inz_dim;
//
//         for (int jx = 0; jx < ix + 1; jx++) {
//           int prod = 1;
//           for (int kx = jx + 1; kx < nx; kx++) {
//             prod *= order_list[kx];
//           }
//           inz_dim = rem / prod;
//           rem -= inz_dim * prod;
//         }
//
//         int inz = i * nnz_row + inz_row;
//
//         int prod = 1;
//         for (int jx = ix + 1; jx < nx; jx++) {
//           prod *= ncp_list[jx];
//         }
//
//         data[inz] *= basis_vec[inz_dim];
//         cols[inz] += (inz_dim + istart) * prod;
//       }
//     }
//   }
// }
