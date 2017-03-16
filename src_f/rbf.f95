subroutine compute_jac(kx, poly, nx, ne, nb, nj, d0, xe, xb, jac)

  implicit none

  !f2py intent(in) kx, poly, nx, ne, nb, nj, d0, xe, xb
  !f2py intent(out) jac
  !f2py depend(nx) d0
  !f2py depend(ne, nx) xe
  !f2py depend(nb, nx) xb
  !f2py depend(ne, nj) jac

  ! Input
  integer, intent(in) :: kx, poly, nx, ne, nb, nj
  double precision, intent(in) :: d0(nx), xe(ne, nx), xb(nb, nx)

  ! Output
  double precision, intent(out) :: jac(ne, nj)

  ! Working
  integer :: ie, ib
  double precision :: rsq, rsq_x
  double precision :: d(nx)
  double precision :: basis, basis_r

  if (kx .eq. 0) then
    do ie = 1, ne
       do ib = 1, nb
          d = xe(ie, :) - xb(ib, :)
          rsq = dot_product(d/d0, d/d0)
          jac(ie, ib) = basis(rsq)
       end do
    end do
    if (poly .ge. 0) then
       jac(:, nb + 1) = 1.0
    end if
    if (poly .eq. 1) then
       jac(:, nb+2:) = xe
    end if
  else
    do ie = 1, ne
       do ib = 1, nb
          d = xe(ie, :) - xb(ib, :)
          rsq = dot_product(d/d0, d/d0)
          rsq_x = 2. * d(kx) / (d0(kx) * d0(kx))
          jac(ie, ib) = basis_r(rsq) * rsq_x
       end do
    end do
    if (poly .ge. 0) then
       jac(:, nb + 1) = 0.0
    end if
    if (poly .eq. 1) then
       jac(:, nb+2:) = 0.0
       jac(:, nb+1+kx) = 1.0
    end if
  end if

end subroutine compute_jac



function basis(rsq)

  implicit none

  double precision, intent(in) :: rsq
  double precision :: basis

  basis = exp(-rsq)

end function basis



function basis_r(rsq)

  implicit none

  double precision, intent(in) :: rsq
  double precision :: basis_r

  basis_r = -exp(-rsq)

end function basis_r
