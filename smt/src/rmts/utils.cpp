#include "utils.hpp"

void expand_index(int nx, int * nlist, int index, int * ilist) {
  int rem = index;

  for (int ix = 0; ix < nx; ix++) {
    int prod = 1;
    for (int kx = ix + 1; kx < nx; kx++) {
      prod *= nlist[kx];
    }
    ilist[ix] = rem / prod;
    rem -= ilist[ix] * prod;
  }
}

int contract_index(int nx, int * nlist, int * ilist) {
  int index = 0;
  int prod = 1;

  for (int ix = nx - 1; ix >= 0; ix--) {
    index += ilist[ix] * prod;
    prod *= nlist[ix];
  }

  return index;
}
