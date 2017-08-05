#include "rmtb.hpp"
#include "rmts.hpp"
#include "utils.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cstring>

using namespace std;

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

  this->order_list = new int[nx];
  this->ncp_list = new int[nx];

  memcpy(this->order_list, order_list, nx * sizeof(*order_list));
  memcpy(this->ncp_list, ncp_list, nx * sizeof(*ncp_list));
}

// void RMTS::compute_ext_dist(int n, int nterm, double * x, double * dx) {
//   double work;
//
//   for (int i = 0; i < n; i++) {
//     for (int ix = 0; ix < nx; ix++) {
//       // compute the vector pointing from the nearest point in the domain to the current x
//       work = x[i * nx + ix];
//       work = max(lower[ix], work);
//       work = min(upper[ix], work);
//       work = x[i * nx + ix] - work;
//       for (int iterm = 0; iterm < nterm; iterm++) {
//         dx[i * nterm * nx + iterm * nx + ix] = work;
//       }
//     }
//   }
// }
//
// void RMTS::compute_quadrature_points(int n, int * nelem_list, double * x) {
//   int ielem_list[nx];
//   double t;
//
//   for (int i = 0; i < n; i++) {
//     expand_index(nx, nelem_list, i, ielem_list);
//     for (int ix = 0; ix < nx; ix++) {
//       t = (1. + 2. * ielem_list[ix]) / 2. / nelem_list[ix];
//       x[i * nx + ix] = lower[ix] + t * (upper[ix] - lower[ix]);
//     }
//   }
// }
