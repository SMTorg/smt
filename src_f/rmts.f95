subroutine compute_ext_dist(nx, neval, ndx, xlimits, xeval, dx)

  implicit none

  !f2py intent(in) nx, neval, ndx, xlimits, xeval
  !f2py intent(out) dx
  !f2py depend(nx) xlimits
  !f2py depend(neval, nx) xeval
  !f2py depend(ndx, nx) dx

  ! Input
  integer, intent(in) :: nx, neval, ndx
  double precision, intent(in) :: xlimits(nx, 2), xeval(neval, nx)

  ! Output
  double precision, intent(out) :: dx(ndx, nx)

  ! Working
  integer :: ieval, ix, iterm, nterm, index
  double precision :: work(neval, nx)

  work(:, :) = xeval(:, :)

  nterm = ndx / neval

  ! After these 2 do loops, work is the position vector from the
  ! nearest point in the domain specified by xlimits.
  do ieval = 1, neval
     do ix = 1, nx
        work = xeval(ieval, ix)
        work = max(xlimits(ix, 1), work)
        work = min(xlimits(ix, 2), work)
        work = xeval(ieval, ix) - work
        do iterm = 1, nterm
           index = (ieval - 1) * nterm + iterm
           dx(index, ix) = work(ieval, ix)
        end do
     end do
  end do

end subroutine compute_ext_dist



subroutine compute_quadrature_points(n, nx, nelem_list, xlimits, x)

  implicit none

  !f2py intent(in) n, nx, nelem_list, xlimits
  !f2py intent(out) x
  !f2py depend(nx) nelem_list, xlimits
  !f2py depend(n, nx) x

  ! Input
  integer, intent(in) :: n, nx, nelem_list(nx)
  double precision, intent(in) :: xlimits(nx, 2)

  ! Output
  double precision, intent(out) :: x(n, nx)

  ! Working
  integer :: i, ix, ielem_list(nx)
  double precision :: t

  do i = 1, n
    call expandindex(nx, nelem_list, i, ielem_list)
    do ix = 1, nx
      t = (-1. + 2 * ielem_list(ix)) / 2. / nelem_list(ix)
      x(i, ix) = xlimits(ix, 1) + t * (xlimits(ix, 2) - xlimits(ix, 1))
    end do
  end do

end subroutine compute_quadrature_points



subroutine compute_quadrature_points_corners(n, nx, npt_list, xlimits, x)

  implicit none

  !f2py intent(in) n, nx, npt_list, xlimits
  !f2py intent(out) x
  !f2py depend(nx) npt_list, xlimits
  !f2py depend(n, nx) x

  ! Input
  integer, intent(in) :: n, nx, npt_list(nx)
  double precision, intent(in) :: xlimits(nx, 2)

  ! Output
  double precision, intent(out) :: x(n, nx)

  ! Working
  integer :: i, ix, ipt_list(nx)
  double precision :: t

  do i = 1, n
    call expandindex(nx, npt_list, i, ipt_list)
    do ix = 1, nx
      t = 1.0 * (ipt_list(ix) - 1) / (npt_list(ix) - 1)
      x(i, ix) = xlimits(ix, 1) + t * (xlimits(ix, 2) - xlimits(ix, 1))
    end do
  end do

end subroutine compute_quadrature_points_corners
