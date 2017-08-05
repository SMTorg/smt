#ifndef __RMTC__
#define __RMTC__

#include "rmts.hpp"
#include <iostream>

class RMTC : public RMTS {
public:
  RMTC();
  ~RMTC();
  void setup(int nx, double * lower, double * upper, int * elem_list, int * term_list);
  // void compute_jac(int ix1, int ix2, int n, double * t, double * data, int * rows, int * cols);

private:
  int * elem_list;
  int * term_list;
};

#endif
