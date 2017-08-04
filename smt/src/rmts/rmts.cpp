#include "rmts.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cstring>

using namespace std;

RMTS::RMTS() {
  lower = NULL;
  upper = NULL;
}

RMTS::~RMTS() {
  delete[] lower;
  delete[] upper;
}

void RMTS::setup(int nx, double * lower, double * upper) {
  this->nx = nx;
  this->lower = new double[nx];
  this->upper = new double[nx];

  memcpy(this->lower, lower, nx * sizeof(*lower));
  memcpy(this->upper, upper, nx * sizeof(*upper));
}

void RMTS::compute_ext_dist(int n, int nterm, double * x, double * dx) {
  double work;

  for (int i = 0; i < n; i++) {
    for (int ix = 0; ix < nx; ix++) {
      // compute the vector pointing from the nearest point in the domain to the current x
      work = x[i * nx + ix];
      work = max(lower[ix], work);
      work = min(upper[ix], work);
      work = x[i * nx + ix] - work;
      for (int iterm = 0; iterm < nterm; iterm++) {
        dx[i * nterm * nx + iterm * nx + ix] = work;
      }
    }
  }
}

// do ieval = 1, neval
//    do ix = 1, nx
//       work = xeval(ieval, ix)
//       work = max(xlimits(ix, 1), work)
//       work = min(xlimits(ix, 2), work)
//       work = xeval(ieval, ix) - work
//       do iterm = 1, nterm
//          index = (ieval - 1) * nterm + iterm
//          dx(index, ix) = work(ieval, ix)
//       end do
//    end do
// end do

// void RMTS::compute_jac(int n, double* x, double* jac) {
//   double w[nt], r2[nt], min_val, sum, d;
//   int min_loc;
//
//   for (int i = 0; i < n; i++) {
//     min_val = 1.;
//     min_loc = 0;
//     for (int it = 0; it < nt; it++) {
//       r2[it] = 0.;
//       for (int ix = 0; ix < nx; ix++) {
//         d = x[i * nx + ix] - xt[it * nx + ix];
//         r2[it] += pow(d, 2);
//       }
//       if (r2[it] < min_val) {
//         min_val = r2[it];
//         min_loc = it;
//       }
//       w[it] = pow(r2[it], -p / 2.);
//     }
//
//     if (min_val == 0.) {
//       for (int it = 0; it < nt; it++) {
//         jac[i * nt + it] = 0.;
//       }
//       jac[i * nt + min_loc] = 1.;
//     }
//     else {
//       sum = 0;
//       for (int it = 0; it < nt; it++) {
//         sum += w[it];
//       }
//       for (int it = 0; it < nt; it++) {
//         jac[i * nt + it] = w[it] / sum;
//       }
//     }
//   }
// }
