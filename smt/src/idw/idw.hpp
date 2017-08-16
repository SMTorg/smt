#include <iostream>

class IDW {
public:
  IDW();
  ~IDW();
  void setup(int nx, int nt, double p, double * xt);
  void compute_jac(int n, double * x, double * jac);
  void compute_jac_derivs(int n, int kx, double* x, double* jac);

private:
  int nx, nt;
  double p;
  double * xt;
  double * w;
  double * dw_dx;
};
