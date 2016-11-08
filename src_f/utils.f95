subroutine expandindex(ndim, nums, index, inds)

  implicit none

  ! Input
  integer, intent(in) :: ndim, nums(ndim), index

  ! Output
  integer, intent(out) :: inds(ndim)

  ! Working
  integer :: idim, rem, prod

  rem = index - 1
  do idim = ndim, 1, -1
     prod = product(nums(:idim-1))
     inds(idim) = rem / prod + 1
     rem = rem - (inds(idim) - 1) * prod
  end do
  
end subroutine expandindex


subroutine contractindex(ndim, nums, inds, index)

  implicit none

  ! Input
  integer, intent(in) :: ndim, nums(ndim), inds(ndim)

  ! Output
  integer, intent(out) :: index

  ! Working
  integer :: idim, prod

  index = 1
  prod = 1
  do idim = 1, ndim
     index = index + (inds(idim)-1) * prod
     prod = prod * nums(idim)
  end do
  
end subroutine contractindex


subroutine uniqueindex(ielem, ideg, index)

  implicit none

  ! Input
  integer, intent(in) :: ielem, ideg

  ! Output
  integer, intent(out) :: index

  index = (ielem-1) * 2 + ideg
  
end subroutine uniqueindex


subroutine findinterval(num, xlimits, x, index, xbar)

  implicit none

  !f2py intent(in) num, xlimits, x
  !f2py intent(out) index, xbar

  ! Input
  integer, intent(in) :: num
  double precision, intent(in) :: xlimits(2), x

  ! Output
  integer, intent(out) :: index
  double precision, intent(out) :: xbar

  ! Working
  double precision :: a, b, a2, b2, bma_d2, apb_d2

  a = xlimits(1)
  b = xlimits(2)
  index = ceiling( (x - a) / (b - a) * num )
  index = max(1, index)
  index = min(num, index)

  a2 = a + (b-a) * (index-1) / num
  b2 = a + (b-a) * (index) / num

  bma_d2 = (b2-a2)/2.
  apb_d2 = (a2+b2)/2.
  xbar = (x - apb_d2) / bma_d2
  xbar = max(-1., xbar)
  xbar = min( 1., xbar)

end subroutine findinterval


subroutine findintervalc(num, xlimits, x, index, xbar)

  implicit none

  !f2py intent(in) num, xlimits, x
  !f2py intent(out) index, xbar

  ! Input
  integer, intent(in) :: num
  double precision, intent(in) :: xlimits(2)
  complex*16, intent(in) :: x

  ! Output
  integer, intent(out) :: index
  complex*16, intent(out) :: xbar

  ! Working
  double precision :: a, b, a2, b2, bma_d2, apb_d2
  complex*16 :: max_rc, min_rc

  a = xlimits(1)
  b = xlimits(2)
  index = ceiling( (real(x) - a) / (b - a) * num )
  index = max(1, index)
  index = min(num, index)

  a2 = a + (b-a) * (index-1) / num
  b2 = a + (b-a) * (index) / num

  bma_d2 = (b2-a2)/2.
  apb_d2 = (a2+b2)/2.
  xbar = (x - apb_d2) / bma_d2
  xbar = max_rc(dble(-1.), xbar)
  xbar = min_rc(dble( 1.), xbar)
!  print *, 'ccccc', x, xbar !!!!!!!! ADDING THIS PRINT CHANGES RESULT

end subroutine findintervalc


complex*16 function max_rc(val1, val2)

  implicit none

  double precision, intent(in) :: val1
  complex*16, intent(in) :: val2

  if (val1 > real(val2)) then
     max_rc = cmplx(val1, 0., kind(1.d0))
  else
     max_rc = val2
  endif

  return

end function max_rc


complex*16 function min_rc(val1, val2)

  implicit none

  double precision, intent(in) :: val1
  complex*16, intent(in) :: val2

  if (val1 < real(val2)) then
     min_rc = cmplx(val1, 0., kind(1.d0))
  else
     min_rc = val2
  endif

  return

end function min_rc
