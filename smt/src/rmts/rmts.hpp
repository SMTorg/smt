#include <iostream>

class RMTS {
public:
  RMTS();
  ~RMTS();
  void setup(int nx, double * lower, double * upper);
  void compute_ext_dist(int n, int nterm, double * x, double * dx);
  // void compute_jac(int n, double * x, double * jac);
  // void compute_jac_derivs(int n, int kx, double* x, double* jac);

private:
  int nx;
  double * lower;
  double * upper;
};
