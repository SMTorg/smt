#include "rbf.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cstring>

using namespace std;

RBF::RBF() {
  d0 = NULL;
  xt = NULL;
}

RBF::~RBF() {
  delete[] d0;
  delete[] xt;
}

void RBF::setup(int nx, int nt, int num_dof, int poly_degree, double * d0, double * xt) {
  delete[] this->d0;
  delete[] this->xt;

  this->nx = nx;
  this->nt = nt;
  this->num_dof = num_dof;
  this->poly_degree = poly_degree;
  this->d0 = new double[nx];
  this->xt = new double[nt * nx];

  memcpy(this->d0, d0, nx * sizeof(*d0));
  memcpy(this->xt, xt, nt * nx * sizeof(*xt));
}

void RBF::compute_jac(int n, double* x, double* jac) {
  double r2, d;

  for (int i = 0; i < n; i++) {
    for (int it = 0; it < nt; it++) {
      r2 = 0.;
      for (int ix = 0; ix < nx; ix++) {
        d = x[i * nx + ix] - xt[it * nx + ix];
        r2 += pow(d / d0[ix], 2);
      }

      jac[i * num_dof + it] = exp(-r2);
    }
  }

  if (poly_degree >= 0) {
    for (int i = 0; i < n; i++) {
      jac[i * num_dof + nt] = 1.;
    }
  }

  if (poly_degree == 1) {
    for (int i = 0; i < n; i++) {
      for (int ix = 0; ix < nx; ix++) {
        jac[i * num_dof + nt + 1 + ix] = x[i * nx + ix];
      }
    }
  }
}

void RBF::compute_jac_derivs(int n, int kx, double* x, double* jac) {
  double r2, dr2_dx, d;

  for (int i = 0; i < n; i++) {
    for (int it = 0; it < nt; it++) {
      r2 = 0.;
      for (int ix = 0; ix < nx; ix++) {
        d = x[i * nx + ix] - xt[it * nx + ix];
        r2 += pow(d / d0[ix], 2);
      }

      d = x[i * nx + kx] - xt[it * nx + kx];
      dr2_dx = 2. * d / pow(d0[kx], 2);

      jac[i * num_dof + it] = -exp(-r2) * dr2_dx;
    }
  }

  if (poly_degree >= 0) {
    for (int i = 0; i < n; i++) {
      jac[i * num_dof + nt] = 0.;
    }
  }

  if (poly_degree == 1) {
    for (int i = 0; i < n; i++) {
      for (int ix = 0; ix < nx; ix++) {
        jac[i * num_dof + nt + 1 + ix] = 0.;
      }
      jac[i * num_dof + nt + 1 + kx] = 1.;
    }
  }
}
