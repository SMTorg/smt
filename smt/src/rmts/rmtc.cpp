#include "rmtc.hpp"
#include "rmts.hpp"
#include "utils.hpp"
#include <assert.h>
#include <math.h>
#include <iostream>
#include <cstring>

using namespace std;

RMTC::RMTC() {
  nelem_list = NULL;
  nterm_list = NULL;
}

RMTC::~RMTC() {
  delete[] nelem_list;
  delete[] nterm_list;
}

void RMTC::find_interval(int ix, int num, double x, int * index, double * xbar) {
  double a = lower[ix];
  double b = upper[ix];

  *index = (int) ceil( (x - a) / (b - a) * num );
  *index = max(1, *index);
  *index = min(num, *index);

  double a2 = a + (b - a) * (*index - 1) / num;
  double b2 = a + (b - a) * (*index    ) / num;

  double bma_d2 = (b2 - a2) / 2.;
  double apb_d2 = (a2 + b2) / 2.;

  *xbar = (x - apb_d2) / bma_d2;
  *xbar = max(-1., *xbar);
  *xbar = min( 1., *xbar);

  *index -= 1;
}

void RMTC::setup(int nx, double * lower, double * upper, int * nelem_list, int * nterm_list) {
  RMTS::setup(nx, lower, upper);

  this->nelem_list = new int[nx];
  this->nterm_list = new int[nx];

  memcpy(this->nelem_list, nelem_list, nx * sizeof(*nelem_list));
  memcpy(this->nterm_list, nterm_list, nx * sizeof(*nterm_list));

  nelem = 1;
  nterm = 1;
  for (int ix = 0; ix < nx; ix++) {
    nelem *= nelem_list[ix];
    nterm *= nterm_list[ix];
  }
}

void RMTC::compute_coeff2nodal(double * mtx) {
  bool deriv_list[4] = { false , false , true , true };
  double xval_list[4] = { -1. , 1. , -1. , 1. };

  for (int iterm1 = 0; iterm1 < nterm; iterm1++) {
    int iterm1_list[nx];
    expand_index(nx, nterm_list, iterm1, iterm1_list);

    for (int iterm2 = 0; iterm2 < nterm; iterm2++) {
      int iterm2_list[nx];
      expand_index(nx, nterm_list, iterm2, iterm2_list);

      int prod = 1;

      for (int ix = 0; ix < nx; ix++) {
        bool deriv = deriv_list[ iterm1_list[ix] ];
        double xval = xval_list[ iterm1_list[ix] ];

        int power = iterm2_list[ix];
        if (deriv) {
          if (power >= 1) {
            prod *= power * pow(xval, power - 1);
          } else {
            prod = 0;
          }
        } else {
          prod *= pow(xval, power);
        }

        mtx[iterm1 * nterm + iterm2] = prod;
      }
    }
  }
}

void RMTC::compute_uniq2elem(double * data, int * rows, int * cols) {
  int derv_map[4] = { 0 , 0 , 1 , 1 };
  int side_map[4] = { 0 , 1 , 0 , 1 };

  int ndofs_list[nx];
  int nuniq_list[nx];
  int ndofs = 1;
  int nuniq = 1;
  for (int ix = 0; ix < nx; ix++) {
    ndofs_list[ix] = 2;
    nuniq_list[ix] = 1 + nelem_list[ix];
    ndofs *= ndofs_list[ix];
    nuniq *= nuniq_list[ix];
  }

  int inz = 0;

  for (int ielem = 0; ielem < nelem; ielem++) {
    int ielem_list[nx];
    expand_index(nx, nelem_list, ielem, ielem_list);

    for (int iterm = 0; iterm < nterm; iterm++) {
      int iterm_list[nx];
      expand_index(nx, nterm_list, iterm, iterm_list);

      int iderv_list[nx];
      int iside_list[nx];
      int iuniq_list[nx];
      int iderv, iuniq;

      for (int ix = 0; ix < nx; ix++) {
        iderv_list[ix] = derv_map[ iterm_list[ix] ];
        iside_list[ix] = side_map[ iterm_list[ix] ];
        iuniq_list[ix] = ielem_list[ix] + iside_list[ix];
      }

      iderv = contract_index(nx, ndofs_list, iderv_list);
      iuniq = contract_index(nx, nuniq_list, iuniq_list);

      data[inz] = 1.;
      rows[inz] = ielem * nterm + iterm;
      cols[inz] = iderv * nuniq + iuniq;
      inz += 1;
    }
  }
}

void RMTC::compute_full_from_block(double * mtx, double * data, int * rows, int * cols){
  int inz = 0;

  for (int ielem = 0; ielem < nelem; ielem++) {
    for (int iterm1 = 0; iterm1 < nterm; iterm1++) {
      for (int iterm2 = 0; iterm2 < nterm; iterm2++) {
        data[inz] = mtx[iterm1 * nterm + iterm2];
        rows[inz] = ielem * nterm + iterm1;
        cols[inz] = ielem * nterm + iterm2;
        inz += 1;
      }
    }
  }
}

void RMTC::compute_jac(
    int ix1, int ix2, int n, double * x,
    double * data, int * rows, int * cols) {

  double dxb_dx[nx];

  for (int ix = 0; ix < nx; ix++) {
    double bma_d2 = (upper[ix] - lower[ix]) / nelem_list[ix] / 2.;
    dxb_dx[ix] = 1. / bma_d2;
  }

  int inz = 0;

  for (int i = 0; i < n; i++) {
    int ielem_list[nx];
    double xbar[nx];

    for (int ix = 0; ix < nx; ix++) {
      find_interval(
        ix, nelem_list[ix], x[i * nx + ix],
        &(ielem_list[ix]), &(xbar[ix]));
    }
    int ielem = contract_index(nx, nelem_list, ielem_list);

    for (int iterm = 0; iterm < nterm; iterm++) {
      int iterm_list[nx];
      expand_index(nx, nterm_list, iterm, iterm_list);

      double prod = 1.;
      for (int ix = 0; ix < nx; ix++) {
        int power = iterm_list[ix];
        if ((ix != ix1) && (ix != ix2)) {
          prod *= pow(xbar[ix], power);
        } else if ((ix == ix1) && (ix == ix2)) {
          if (power >= 2) {
            prod *= power * (power - 1) * pow(xbar[ix], power - 2) * dxb_dx[ix] * dxb_dx[ix];
          } else {
            prod = 0.;
          }
        } else {
          if (power >= 1) {
            prod *= power * pow(xbar[ix], power - 1) * dxb_dx[ix];
          } else {
            prod = 0.;
          }
        }
      }

      data[inz] = prod;
      rows[inz] = i;
      cols[inz] = ielem * nterm + iterm;
      inz += 1;
    }
  }
}
