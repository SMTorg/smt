#ifndef __RMTC__
#define __RMTC__

#include "rmts.hpp"
#include <iostream>

class RMTC : public RMTS {
public:
  RMTC();
  ~RMTC();
  void setup(int nx, double * lower, double * upper, int * nelem_list, int * nterm_list);
  void compute_coeff2nodal(double * mtx);
  void compute_uniq2elem(double * data, int * rows, int * cols);
  void compute_full_from_block(double * mtx, double * data, int * rows, int * cols);
  void compute_jac(int ix1, int ix2, int n, double * x, double * data, int * rows, int * cols);

private:
  void find_interval(int ix, int num, double x, int * index, double * xbar);
  int * nelem_list;
  int * nterm_list;
  long nelem, nterm;
};

#endif
