#include "rmts.hpp"
#include "utils.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

RMTS::RMTS() {
  lower = NULL;
  upper = NULL;

  work_int_nx_1 = NULL;
  work_int_nx_2 = NULL;
  work_int_nx_3 = NULL;
  work_int_nx_4 = NULL;
  work_int_nx_5 = NULL;
  work_int_nx_6 = NULL;
  work_int_nx_7 = NULL;

  work_double_nx_1 = NULL;
  work_double_nx_2 = NULL;
}

RMTS::~RMTS() {
  delete[] lower;
  delete[] upper;

  delete[] work_int_nx_1;
  delete[] work_int_nx_2;
  delete[] work_int_nx_3;
  delete[] work_int_nx_4;
  delete[] work_int_nx_5;
  delete[] work_int_nx_6;
  delete[] work_int_nx_7;

  delete[] work_double_nx_1;
  delete[] work_double_nx_2;
}

void RMTS::setup(int nx, double * lower, double * upper) {
  delete[] this->lower;
  delete[] this->upper;

  delete[] this->work_int_nx_1;
  delete[] this->work_int_nx_2;
  delete[] this->work_int_nx_3;
  delete[] this->work_int_nx_4;
  delete[] this->work_int_nx_5;
  delete[] this->work_int_nx_6;
  delete[] this->work_int_nx_7;

  delete[] this->work_double_nx_1;
  delete[] this->work_double_nx_2;

  this->nx = nx;
  this->lower = new double[nx];
  this->upper = new double[nx];

  this->work_int_nx_1 = new int[nx];
  this->work_int_nx_2 = new int[nx];
  this->work_int_nx_3 = new int[nx];
  this->work_int_nx_4 = new int[nx];
  this->work_int_nx_5 = new int[nx];
  this->work_int_nx_6 = new int[nx];
  this->work_int_nx_7 = new int[nx];

  this->work_double_nx_1 = new double[nx];
  this->work_double_nx_2 = new double[nx];

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

void RMTS::compute_quadrature_points(int n, int * nelem_list, double * x) {
  int * ielem_list;
  ielem_list = work_int_nx_1;

  double t;

  for (int i = 0; i < n; i++) {
    expand_index(nx, nelem_list, i, ielem_list);
    for (int ix = 0; ix < nx; ix++) {
      t = (1. + 2. * ielem_list[ix]) / 2. / nelem_list[ix];
      x[i * nx + ix] = lower[ix] + t * (upper[ix] - lower[ix]);
    }
  }
}
