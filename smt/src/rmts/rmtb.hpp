#ifndef __RMTB__
#define __RMTB__

#include "rmts.hpp"
#include <iostream>

class RMTB : public RMTS {
public:
  RMTB();
  ~RMTB();
  void setup(int nx, double * lower, double * upper, int * order_list, int * ncp_list);

private:
  int * order_list;
  int * ncp_list;
};

#endif
