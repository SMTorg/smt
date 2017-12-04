#include "idw.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cstring>

using namespace std;

IDW::IDW() {
  xt = NULL;
  w = NULL;
  dw_dx = NULL;
}

IDW::~IDW() {
  delete[] xt;
  delete[] w;
  delete[] dw_dx;
}

void IDW::setup(int nx, int nt, double p, double * xt) {
  delete[] this->xt;
  delete[] this->w;
  delete[] this->dw_dx;

  this->nx = nx;
  this->nt = nt;
  this->p = p;

  this->xt = new double[nt * nx];
  this->w = new double[nt];
  this->dw_dx = new double[nt];

  memcpy(this->xt, xt, nt * nx * sizeof(*xt));
}

void IDW::compute_jac(int n, double* x, double* jac) {
  double r2, min_val, sum, d;
  int min_loc;

  for (int i = 0; i < n; i++) {
    min_val = 1.;
    min_loc = 0;
    for (int it = 0; it < nt; it++) {
      r2 = 0.;
      for (int ix = 0; ix < nx; ix++) {
        d = x[i * nx + ix] - xt[it * nx + ix];
        r2 += pow(d, 2);
      }
      if (r2 < min_val) {
        min_val = r2;
        min_loc = it;
      }
      w[it] = pow(r2, -p / 2.);
    }

    if (min_val == 0.) {
      for (int it = 0; it < nt; it++) {
        jac[i * nt + it] = 0.;
      }
      jac[i * nt + min_loc] = 1.;
    }
    else {
      sum = 0;
      for (int it = 0; it < nt; it++) {
        sum += w[it];
      }
      for (int it = 0; it < nt; it++) {
        jac[i * nt + it] = w[it] / sum;
      }
    }
  }
}

void IDW::compute_jac_derivs(int n, int kx, double* x, double* jac) {
  double r2, dr2_dx, min_val, sum, dsum_dx, d;
  int min_loc;

  for (int i = 0; i < n; i++) {
    min_val = 1.;
    min_loc = 0;
    for (int it = 0; it < nt; it++) {
      r2 = 0.;
      for (int ix = 0; ix < nx; ix++) {
        d = x[i * nx + ix] - xt[it * nx + ix];
        r2 += pow(d, 2);
      }
      d = x[i * nx + kx] - xt[it * nx + kx];
      dr2_dx = 2. * d;

      if (r2 < min_val) {
        min_val = r2;
        min_loc = it;
      }
      w[it] = pow(r2, -p / 2.);
      dw_dx[it] = -p / 2. * pow(r2, -p / 2. - 1.) * dr2_dx;
    }

    if (min_val == 0.) {
      for (int it = 0; it < nt; it++) {
        jac[i * nt + it] = 0.;
      }
    }
    else {
      sum = 0;
      dsum_dx = 0;
      for (int it = 0; it < nt; it++) {
        sum += w[it];
        dsum_dx += dw_dx[it];
      }
      for (int it = 0; it < nt; it++) {
        // jac[i * nt + it] = w[it] / sum;
        jac[i * nt + it] = (dw_dx[it] * sum - w[it] * dsum_dx) / pow(sum, 2);
      }
    }
  }
}
