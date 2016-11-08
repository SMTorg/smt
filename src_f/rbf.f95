subroutine rbfgetjac(poly, nx, ne, nb, nj, d0, xe, xb, jac)

  implicit none

  !f2py intent(in)   poly, nx, ne, nb, nj, d0, xe, xb
  !f2py intent(out) jac
  !f2py depend(nx) d0
  !f2py depend(ne, nx) xe
  !f2py depend(nb, nx) xb
  !f2py depend(ne, nj) jac

  ! Input
  integer, intent(in) :: poly, nx, ne, nb, nj
  double precision, intent(in) :: d0(nx), xe(ne, nx), xb(nb, nx)

  ! Output
  double precision, intent(out) :: jac(ne, nj)

  ! Working
  integer :: ie, ib
  double precision :: rsq
  double precision :: d(nx)

  do ie = 1, ne
     do ib = 1, nb
        d = xe(ie, :) - xb(ib, :)
        rsq = dot_product(d/d0, d/d0)
        jac(ie, ib) = exp(-rsq)
     end do
  end do
  if (poly .eq. 1) then
     jac(:, nb + 1) = 1.0
  end if

  if (poly .eq. 2) then
     jac(:, nb+1:) = xe
  end if

end subroutine rbfgetjac
