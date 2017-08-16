#ifndef __RMTS__
#define __RMTS__

#include <iostream>

class RMTS {
public:
  RMTS();
  ~RMTS();
  void setup(int nx, double * lower, double * upper);
  void compute_ext_dist(int n, int nterm, double * x, double * dx);
  void compute_quadrature_points(int n, int * nelem_list, double * x);

protected:
  int nx;
  double * lower;
  double * upper;
  int * work_int_nx_1;
  int * work_int_nx_2;
  int * work_int_nx_3;
  int * work_int_nx_4;
  int * work_int_nx_5;
  int * work_int_nx_6;
  int * work_int_nx_7;
  double * work_double_nx_1;
  double * work_double_nx_2;
};

#endif
