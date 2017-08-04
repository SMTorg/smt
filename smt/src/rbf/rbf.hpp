#include <iostream>

class RBF {
public:
  RBF();
  ~RBF();
  void setup(int nx, int nt, int num_dof, int poly_degree, double * d0, double * xt);
  void compute_jac(int n, double * x, double * jac);
  void compute_jac_derivs(int n, int kx, double* x, double* jac);

private:
  int nx, nt, num_dof;
  int poly_degree;
  double * d0;
  double * xt;
};
