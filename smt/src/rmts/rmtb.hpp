#ifndef __RMTB__
#define __RMTB__

#include "rmts.hpp"
#include <iostream>

class RMTB : public RMTS {
public:
  RMTB();
  ~RMTB();
  void setup(int nx, double * lower, double * upper, int * order_list, int * ncp_list);
  void compute_jac(int ix1, int ix2, int n, double * t, double * data, int * rows, int * cols);

private:
  int * order_list;
  int * ncp_list;
};

#endif
