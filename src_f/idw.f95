subroutine compute_jac(nx, ne, nt, power, xe, xt, jac)

  implicit none

  !f2py intent(in) nx, ne, nt, power, xe, xt
  !f2py intent(out) jac
  !f2py depend(ne, nx) xe
  !f2py depend(nt, nx) xt
  !f2py depend(ne, nt) jac

  ! Input
  integer, intent(in) :: nx, ne, nt
  double precision, intent(in) :: power, xe(ne, nx), xt(nt, nx)

  ! Output
  double precision, intent(out) :: jac(ne, nt)

  ! Working
  integer :: ie, it
  double precision :: sq_dist(nt), d(nx)

  do ie = 1, ne
     do it = 1, nt
        d = xe(ie, :) - xt(it, :)
        sq_dist(it) = dot_product(d, d)
     end do

     if (minval(sq_dist) .eq. 0) then
        jac(ie, :) = 0.0
        jac(ie, minloc(sq_dist)) = 1.0
     else
        jac(ie, :) = sq_dist ** (-power / 2.)
        jac(ie, :) = jac(ie, :) / sum(jac(ie, :))
     end if
  end do

end subroutine compute_jac
